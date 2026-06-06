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
from common import bin_image, find_eclipse, remove_epoch
from toi519_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from toi519_run import get_model, PLANET as PLANET_PARAMS, RADIUS_SCALE_MIN, RADIUS_SCALE_MAX
from toi519_radius2d_null import FIGSIZE

PREFIX = 'toi519'
IC = 'BIC'
TRUE_TEMPERATURE_RATIO = 0.5
TRUE_LOG_EPSILON = temp_to_log_epsilon([TRUE_TEMPERATURE_RATIO])
MAX_BASIS = None
NOISE_SCALE = 1.0
CHI2_NOISE_SCALE = np.sqrt(11.695001707659017*1.0641359591613742)
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
    data = VSPEC.PhaseAnalyzer.from_model(get_model())
    wl = data.wavelength
    time = data.time
    
    test_thermal = THERMAL_SCALE*interpolator([1])[0, :, :].T
    logger.info(f'Thermal has shape {test_thermal.shape}.')
    eclipse_start, eclipse_end = find_eclipse(test_thermal[:, -1])
    logger.info(f'Eclipse index is {eclipse_start} to {eclipse_end}.')
    
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
            remove_epoch(_total_observed,eclipse_start,eclipse_end),
            remove_epoch(_scatter_mag,eclipse_start,eclipse_end),
            _cutoff_index,
            True,
            IC,
            max_basis_size=MAX_BASIS
        )
        _residual = _f_rec - remove_epoch(_total_observed,eclipse_start,eclipse_end)
        return _residual, remove_epoch(_uncertainty,eclipse_start,eclipse_end), _s, _coeffs

    temp_array = np.linspace(0.05, 0.99, 150)
    log_eps_array = (temp_to_log_epsilon(temp_array))
    
    pl_true_radius = PLANET_PARAMS.radius.to(u.R_jup)
    radius_arr = np.linspace(RADIUS_SCALE_MIN, RADIUS_SCALE_MAX,80)
    red_chi_sq_array = np.zeros((radius_arr.size, log_eps_array.size))
    data_residual, data_noise, _s, _coeffs = get_residual_and_noise(
        chi_noise_scale=CHI2_NOISE_SCALE
    )
    # data_residual = remove_epoch(data_residual, eclipse_index)
    # data_noise = remove_epoch(data_noise, eclipse_index)
    data_residual = bin_image(data_residual,BIN_WL,BIN_TIME,1)
    data_noise = bin_image(data_noise,BIN_WL,BIN_TIME,2)
    binned_wl = bin_image(wl.to_value(u.um),BIN_WL,1,1)[0,:]
    for i, rad in tqdm(enumerate(radius_arr),total=radius_arr.size):
        for j, log_eps in enumerate(log_eps_array):
            model_thermal = rad**2*THERMAL_SCALE * \
                interpolator([log_eps])[0, :, :].T
            model_reconstruction = vpie.get_reconstruction(
                model_thermal,
                _coeffs,
                _s
            )
            model_residual = model_reconstruction - remove_epoch(model_thermal, eclipse_start,eclipse_end)
            # model_residual = remove_epoch(model_residual, eclipse_index)
            binned_model_residual = bin_image(model_residual, BIN_WL,BIN_TIME, 1)
            difference = binned_model_residual - data_residual
            chi_sq_spec = difference**2/(data_noise)**2
            long_wl = binned_wl >= CHI2_WL.to_value(u.um)
            chi_sq_spec = chi_sq_spec[:, long_wl]
            chi_sq = np.sum(chi_sq_spec)
            red_chi_sq = chi_sq / (chi_sq_spec.size-2)
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
        ax.text(0.05,0.05,'d) Eclipse ignored',transform=ax.transAxes,fontsize=10,color='w',ha='left',va='center',fontweight='bold')
        ax.scatter(TRUE_TEMPERATURE_RATIO,pl_true_radius.to_value(u.R_jup),marker='*',c='#c50d15',s=200,edgecolor='w')
        fig.tight_layout()
        fig.savefig(
            paths.figures / f'{PREFIX}_retrieval_red_chi_square_radius_half.pdf')
