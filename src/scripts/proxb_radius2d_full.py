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
from tqdm.auto import tqdm

from vpie import vpie
import VSPEC

import paths
from common import bin_image, FIGSIZE, figure_context
from proxb_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from proxb_run import (
    get_model,
    PLANET as PLANET_PARAMS,
    SHORT_WL_CUTOFF as CUTOFF_WL,
    CHI2_WL,
    BIN_WL,
    BIN_TIME,
    RADIUS_SCALE_MIN, RADIUS_SCALE_MAX, TEMP_RATIO_MIN, TEMP_RATIO_MAX
)

PREFIX = 'proxb'
IC = 'BIC'
MAX_BASIS = None
TRUE_TEMPERATURE_RATIO = 0.05
TRUE_LOG_EPSILON = temp_to_log_epsilon([TRUE_TEMPERATURE_RATIO])
NOISE_SCALE = 1.0
CHI2_NOISE_SCALE = np.sqrt(1.7567213198638865)
THERMAL_SCALE = 1.0
SEED = 33
FLUX_UNIT = u.Unit('W m-2 um-1')


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
        _thermal = THERMAL_SCALE*interpolator([np.log10(epsilon)])[0, :, :].T
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

    temp_array = np.linspace(TEMP_RATIO_MIN, TEMP_RATIO_MAX, 150)
    log_eps_array = (temp_to_log_epsilon(temp_array))


    def is_one(x):
        tol = 1e-3
        return (x < 1+tol) and (x > 1-tol)
    pl_true_radius = PLANET_PARAMS.radius.to(u.R_earth)
    radius_arr = np.linspace(RADIUS_SCALE_MIN, RADIUS_SCALE_MAX,80)
    red_chi_sq_array = np.zeros((radius_arr.size, log_eps_array.size))
    chi_sq_eq_nine_array = np.zeros((radius_arr.size, log_eps_array.size))
    dist_residual, dist_noise, _s, _coeffs = get_residual_and_noise(
        epsilon=10**TRUE_LOG_EPSILON,
        chi_noise_scale=np.sqrt(2.27)
    )
    dist_residual = bin_image(dist_residual,BIN_WL,BIN_TIME,1)
    dist_noise = bin_image(dist_noise,BIN_WL,BIN_TIME,2)
    binned_wl = bin_image(wl.to_value(u.um),BIN_WL,1,1)[0,:]
    for i, rad in tqdm(enumerate(radius_arr), total=radius_arr.size):
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
    with figure_context(figsize=FIGSIZE) as fig:
        ax: plt.Axes = fig.subplots(1, 1)
        # ax0.plot(temp_array, best_radius_array.mean(axis=0), c='k')
        # ax0.set_ylabel('$d/\\mathrm{[pc]}$')
        # ax0.set_ylim(0, 3.1)
        # ax0.set_yticks([0, 1, 2, 3])
        # ax0.set_facecolor('w')
        # ax0.axhline(115, c='r', ls='--', zorder=-100)
        im = ax.pcolormesh(
            temp_array, (radius_arr * pl_true_radius).to_value(u.R_earth), (red_chi_sq_array),
            rasterized=True,
            norm=LogNorm(),
            zorder=-100
        )
        ax.set_ylabel('$R_\\mathrm{p}/R_\\oplus$')
        ax.set_xlabel('$T_{\\rm night} / T_{\\rm day}$')
        # ax.set_yscale('log')
        ax.grid(False)
        # yticks = [10, 20, 30, 40, 60, 100, 140]
        # ax.set_yticks(yticks)
        # ax_yticklabels(yticks)
        fig.colorbar(im, label='$\\chi^2_{\\rm red}$')
        levels = [1, 4, 9, 16, 25]
        def fmt(x): return f'$\\chi^2_{{\\rm red}} = {x:.0f}$'
        im = ax.contour(
            temp_array, (radius_arr * pl_true_radius).to_value(u.R_earth), red_chi_sq_array,
            levels=levels,
            colors='k',
            linestyles='dashed'
        )
        ax.clabel(im, im.levels, inline=True, fontsize=10, fmt=fmt)
        levels=[0.75,1,1.3,2.6]
        labels = [ # See Zeng+2019
                   # Also note that Lopez & Fortney (2014) show that insolation is not important in determining the radius
            '$100\\%\\;\\mathrm{Fe}$',
            'Earth-like',
            '50% rock & 50% H$_2$O',
            '+2% H$_2$',
            # '$\\mathrm{Thick\\; H_2\\; Envelope}$'
        ]
        im=ax.contour(
            temp_array,(radius_arr * pl_true_radius).to_value(u.R_earth),np.meshgrid(temp_array,(radius_arr * pl_true_radius).to_value(u.R_earth))[1],
            levels=levels,
            colors='w',
            linestyles='-',
            zorder=-99
        )
        fmt = lambda x: dict(zip(levels, labels))[x]
        # manual = [(0.8,1),(0.3,2.2),(0.2,3)]
        ax.clabel(im,im.levels,inline=True,fontsize=10,fmt=fmt)
        # ax.text(0.5,0.7,'$\\mathrm{Thick\\; H_2\\; Envelope}$',transform=ax.transAxes,fontsize=10,color='w',ha='center',va='center')
        ax.scatter(TRUE_TEMPERATURE_RATIO,pl_true_radius.to_value(u.R_earth),marker='*',c='#c50d15',s=200,edgecolor='w')
        # ax.set_title('Proxima Centauri b', fontsize=16, fontweight='bold')
        fig.tight_layout()
        fig.savefig(
            paths.figures / f'{PREFIX}_retrieval_red_chi_square_radius_full.pdf')
