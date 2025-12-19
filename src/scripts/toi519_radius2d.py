"""
Simplified JWST retrieval script

1. Generate observation

"""

import contextlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy import units as u
from loguru import logger
from scipy.optimize import minimize_scalar

from vpie import vpie
import VSPEC

import paths
from common import bin_image
from toi519_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from toi519_run import get_model, PLANET as PLANET_PARAMS

PREFIX = 'toi519'
IC = 'BIC'
MAX_BASIS = None
TRUE_TEMPERATURE_RATIO = 0.1
TRUE_LOG_EPSILON = temp_to_log_epsilon([TRUE_TEMPERATURE_RATIO])
NOISE_SCALE = 1.0
CHI2_NOISE_SCALE = np.sqrt(2.72)
THERMAL_SCALE = 1.0
SEED = 33
FLUX_UNIT = u.Unit('W m-2 um-1')
CUTOFF_WL = 0.8*u.um
CHI2_WL = 4.0*u.um
BIN_WL = 6
BIN_TIME = 4


@contextlib.contextmanager
def figure_context(*args, **kwargs):
    fig: plt.Figure = plt.figure(*args, **kwargs)
    yield fig
    plt.close(fig)


if __name__ in '__main__':
    plt.style.use('bmh')
    interpolator = get_interp()
    """
    Takes in [log_epsilon]
    """
    thermal = THERMAL_SCALE * \
        interpolator([TRUE_LOG_EPSILON])[0, :, :].T  # m x n
    data = VSPEC.PhaseAnalyzer.from_model(get_model())
    wl = data.wavelength
    time = data.time

    rng = np.random.default_rng(SEED)
    stellar = data.star.T.to_value(FLUX_UNIT)
    true_noise = data.noise.T.to_value(FLUX_UNIT)
    noise = true_noise * NOISE_SCALE
    total_true = stellar + thermal
    scatter = rng.normal(loc=0, scale=noise)
    total_observed = total_true + scatter


    cutoff_index = np.argwhere(wl > CUTOFF_WL)[0][0]
    logger.info(
        f'For a short-wave cutoff of {CUTOFF_WL}, we choose a cutoff index of {cutoff_index}. Total wl axis size is {wl.size}')
    long_cutoff_index = np.argwhere(wl > CHI2_WL)[0][0]
    logger.info(
        f'For a long-wave cutoff of {CHI2_WL}, we choose a cutoff index of {long_cutoff_index}. Total wl axis size is {wl.size}')
    s, coeffs, f_rec = vpie.get_vpie(
        total_observed,
        noise,
        cutoff_index,
        True,
        IC
    )

    residual = f_rec - total_observed

    def get_residual_and_noise(chi_noise_scale=1.0, epsilon=10**TRUE_LOG_EPSILON):
        # logger.info(
        #     f'Running radius={radius:.2f} Rp with noise scale of {chi_noise_scale:.2f}')
        _thermal = 0*THERMAL_SCALE*interpolator([np.log10(epsilon)])[0, :, :].T
        _data = VSPEC.PhaseAnalyzer.from_model(get_model())
        _rng = np.random.default_rng(SEED)
        _stellar = _data.star.T.to_value(FLUX_UNIT)
        _scatter_mag = _data.noise.T.to_value(
            FLUX_UNIT) * NOISE_SCALE
        _total_true = _stellar + _thermal
        _scatter = _rng.normal(loc=0, scale=_scatter_mag)
        _uncertainty = _scatter_mag * chi_noise_scale
        _total_observed = _total_true + _scatter
        _cutoff_index = np.argwhere(wl > CUTOFF_WL)[0][0]
        _s, _coeffs, _f_rec = vpie.get_vpie(
            _total_observed,
            _scatter_mag,
            _cutoff_index,
            True,
            IC,
            max_basis_size=MAX_BASIS
        )
        _residual = _f_rec - _total_observed
        return _residual, _uncertainty, _s, _coeffs

    temp_array = np.linspace(0.05, 0.99, 60)
    log_eps_array = (temp_to_log_epsilon(temp_array))

    temp_ratios = [0.1]
    epsilons = temp_to_log_epsilon(temp_ratios)
    fnames = ['1']
    bounds = (0, 500.0)

    def is_one(x):
        tol = 1e-3
        return (x < 1+tol) and (x > 1-tol)
    for epsilon, fname in zip(epsilons, fnames):
        pl_true_radius = PLANET_PARAMS.radius.to(u.R_jup)
        radius_arr = np.linspace(0.1, 2,40)
        red_chi_sq_array = np.zeros((radius_arr.size, log_eps_array.size))
        chi_sq_eq_nine_array = np.zeros((radius_arr.size, log_eps_array.size))
        dist_residual, dist_noise, _s, _coeffs = get_residual_and_noise(
            epsilon=10**epsilon,
            chi_noise_scale=np.sqrt(2.27)
        )
        dist_residual = bin_image(dist_residual,BIN_WL,BIN_TIME,1)
        dist_noise = bin_image(dist_noise,BIN_WL,BIN_TIME,2)
        binned_wl = bin_image(wl.to_value(u.um),BIN_WL,1,1)[0,:]
        for i, rad in enumerate(radius_arr):
            for j, log_eps in enumerate(log_eps_array):
                grid_thermal = rad**2*THERMAL_SCALE * \
                    interpolator([log_eps])[0, :, :].T
                grid_reconstruction = vpie.get_reconstruction(
                    grid_thermal,
                    _coeffs,
                    _s
                )
                grid_residual = grid_reconstruction - grid_thermal
                binned_grid_residual = bin_image(grid_residual, BIN_WL,BIN_TIME, 1)
                difference = binned_grid_residual - dist_residual
                chi_sq_spec = difference**2/(dist_noise * (CHI2_NOISE_SCALE))**2
                long_wl = binned_wl >= CHI2_WL.to_value(u.um)
                chi_sq_spec = chi_sq_spec[:, long_wl]
                chi_sq = np.sum(chi_sq_spec)
                red_chi_sq = chi_sq / (chi_sq_spec.size+2)
                red_chi_sq_array[i, j] = red_chi_sq
        logger.info(f'The lowest value for red chi2 is {np.min(red_chi_sq_array)}')
        with figure_context(figsize=(6, 4.5)) as fig:
            ax: plt.Axes = fig.subplots(1, 1)
            # ax0.plot(temp_array, best_radius_array.mean(axis=0), c='k')
            # ax0.set_ylabel('$d/\\mathrm{[pc]}$')
            # ax0.set_ylim(0, 3.1)
            # ax0.set_yticks([0, 1, 2, 3])
            # ax0.set_facecolor('w')
            # ax0.axhline(115, c='r', ls='--', zorder=-100)
            im = ax.pcolormesh(
                temp_array, (radius_arr * pl_true_radius).to_value(u.R_jup), (red_chi_sq_array),
                rasterized=True,
                norm=LogNorm()
            )
            ax.set_ylabel('$R_\\mathrm{p}/R_\\mathrm{J}$')
            ax.set_xlabel('$T_{\\rm night} / T_{\\rm day}$')
            # ax.set_yscale('log')
            ax.grid(False)
            # yticks = [10, 20, 30, 40, 60, 100, 140]
            # ax.set_yticks(yticks)
            # ax_yticklabels(yticks)
            ax.axhline(y=pl_true_radius.to_value(u.R_jup), c='r', ls='--')
            fig.colorbar(im, label='$\\chi^2_{\\rm red}$')
            levels = [1, 4, 9, 16, 25]
            def fmt(x): return f'$\\chi^2_{{\\rm red}} = {x:.0f}$'
            im = ax.contour(
                temp_array, radius_arr, red_chi_sq_array,
                levels=levels,
                colors='k',
                linestyles='dashed'
            )
            ax.clabel(im, im.levels, inline=True, fontsize=10, fmt=fmt)
            levels=[20,40,80,160]
            im=ax.contour(
                temp_array,radius_arr,chi_sq_eq_nine_array,
                levels=levels,
                colors='w',
                linestyles='-'
            )
            fmt = lambda x: f'$d = {x:.0f}~\\mathrm{{pc}}$'
            ax.clabel(im,im.levels,inline=True,fontsize=10,fmt=fmt)
            # pos = ax.get_position()
            # pos0 = ax0.get_position()
            # ax0.set_position([pos.x0, pos0.y0, pos.width, pos0.height])
            fig.savefig(
                paths.figures / f'{PREFIX}_retrieval_red_chi_square_radius_{fname}.pdf')
