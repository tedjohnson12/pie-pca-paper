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

from vpie import vpie
import VSPEC

import paths
from common import bin_image
from toi519_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from toi519_run import get_model, PLANET as PLANET_PARAMS
from toi519_radius2d import FIGSIZE

PREFIX = 'toi519'
IC = 'BIC'
TRUE_TEMPERATURE_RATIO = 0.5
TRUE_LOG_EPSILON = temp_to_log_epsilon([TRUE_TEMPERATURE_RATIO])
MAX_BASIS = None
NOISE_SCALE = 1.0
CHI2_NOISE_SCALE = np.sqrt(5.671790795629628)
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

def find_eclipse(a: np.ndarray):
    next = np.concatenate([a[1:], a[-1:]])
    diff = next-a
    return np.argmax(diff)

def remove_epoch(thermal:np.ndarray, index: int):
    return np.concatenate([thermal[:index,:], thermal[index+1:,:]], axis=0)


if __name__ in '__main__':
    plt.style.use('bmh')
    interpolator = get_interp()
    """
    Takes in [log_epsilon]
    """
    data = VSPEC.PhaseAnalyzer.from_model(get_model())
    wl = data.wavelength
    time = data.time
    
    test_thermal = THERMAL_SCALE*interpolator([1])[0, :, :].T
    logger.info(f'Thermal has shape {test_thermal.shape}.')
    eclipse_index = find_eclipse(test_thermal[:, -1])
    logger.info(f'Eclipse index is {eclipse_index}.')
    
    def get_residual_and_noise(chi_noise_scale=1.0):
        # logger.info(
        #     f'Running radius={radius:.2f} Rp with noise scale of {chi_noise_scale:.2f}')
        _thermal = THERMAL_SCALE*interpolator([TRUE_LOG_EPSILON])[0, :, :].T
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
    
    pl_true_radius = PLANET_PARAMS.radius.to(u.R_jup)
    radius_arr = np.linspace(0.1, 2,40)
    red_chi_sq_array = np.zeros((radius_arr.size, log_eps_array.size))
    dist_residual, dist_noise, _s, _coeffs = get_residual_and_noise(
        chi_noise_scale=CHI2_NOISE_SCALE
    )
    dist_residual = remove_epoch(dist_residual, eclipse_index)
    dist_noise = remove_epoch(dist_noise, eclipse_index)
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
            grid_residual = remove_epoch(grid_residual, eclipse_index)
            binned_grid_residual = bin_image(grid_residual, BIN_WL,BIN_TIME, 1)
            difference = binned_grid_residual - dist_residual
            chi_sq_spec = difference**2/(dist_noise)**2
            long_wl = binned_wl >= CHI2_WL.to_value(u.um)
            chi_sq_spec = chi_sq_spec[:, long_wl]
            chi_sq = np.sum(chi_sq_spec)
            red_chi_sq = chi_sq / (chi_sq_spec.size+2)
            red_chi_sq_array[i, j] = red_chi_sq
    logger.info(f'The lowest value for red chi2 is {np.min(red_chi_sq_array)}')
    with figure_context(figsize=FIGSIZE) as fig:
        ax: plt.Axes = fig.subplots(1, 1)
        im = ax.pcolormesh(
            temp_array, (radius_arr * pl_true_radius).to_value(u.R_jup), (red_chi_sq_array),
            rasterized=True,
            norm=LogNorm()
        )
        ax.set_ylabel('$R_\\mathrm{p}/R_\\mathrm{J}$')
        ax.set_xlabel('$T_{\\rm night} / T_{\\rm day}$')
        ax.grid(False)
        # ax.axhline(y=pl_true_radius.to_value(u.R_jup), c='r', ls='--')
        fig.colorbar(im, label='$\\chi^2_{\\rm red}$')
        levels = [1, 4, 9, 16, 25, 100, 225, 400]
        def fmt(x):
            return f'$\\chi^2_{{\\rm red}} = {x:.0f}$'
        im = ax.contour(
            temp_array, radius_arr, red_chi_sq_array,
            levels=levels,
            colors='k',
            linestyles='dashed'
        )
        ax.clabel(im, im.levels, inline=True, fontsize=10, fmt=fmt)
        ax.text(0.05,0.05,'b) Eclipse ignored',transform=ax.transAxes,fontsize=10,color='w',ha='left',va='center',fontweight='bold')
        ax.scatter(TRUE_TEMPERATURE_RATIO,pl_true_radius.to_value(u.R_jup),marker='*',c='w',s=200,edgecolor='k')
        fig.tight_layout()
        fig.savefig(
            paths.figures / f'{PREFIX}_retrieval_red_chi_square_radius_half.pdf')
