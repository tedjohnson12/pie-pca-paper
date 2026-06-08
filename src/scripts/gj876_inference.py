"""
Simplified JWST retrieval script

1. Generate observation

"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy import units as u
from loguru import logger
from tqdm.auto import tqdm

from vpie import vpie
import VSPEC
from vpie import bin_image

import paths
from common import COLWIDTH, figure_context
from gj876_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from gj876_run import (
    get_model, PLANET as PLANET_PARAMS,
    RADIUS_SCALE_MIN, RADIUS_SCALE_MAX,
    TEMP_RATIO_MIN, TEMP_RATIO_MAX
)

PREFIX = 'gj876'
IC = 'BIC'
MAX_BASIS = None
TRUE_TEMPERATURE_RATIO = 0.1
TRUE_LOG_EPSILON = temp_to_log_epsilon([TRUE_TEMPERATURE_RATIO])
NOISE_SCALE = 1.0
THERMAL_SCALE = 1.0
SEED = 33
FLUX_UNIT = u.Unit('W m-2 um-1')
CUTOFF_WL = 0.8*u.um
CHI2_WL = 4.0*u.um
BIN_WL = 6
BIN_TIME = 3

FIGSIZE = (COLWIDTH, 0.7*COLWIDTH)

if __name__ in '__main__':
    plt.style.use('bmh')
    interpolator = get_interp()
    """
    Takes in [log_epsilon]
    """
    data = VSPEC.PhaseAnalyzer.from_model(get_model())
    wl = data.wavelength
    time = data.time
    cutoff_index = np.argwhere(wl > CUTOFF_WL)[0][0]

    def get_residual_and_noise(chi_noise_scale, _epsilon):
        """
        Data from simulated observation
        """
        _thermal = THERMAL_SCALE*interpolator([np.log10(_epsilon)])[0, :, :].T
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
    log_eps_array = temp_to_log_epsilon(temp_array)

    temp_ratios = [0.05,0.5,0.99]
    log_epsilons = temp_to_log_epsilon(temp_ratios)
    fnames = ['none','mod','high']
    set_title = [False,False,True]
    noise_scale = np.sqrt([
        28.191472784293506,
        13.546932306242084,
        17.460894606597723
    ])

    for log_epsilon, fname,_title,_noise_scale,temp_ratio in zip(
        log_epsilons, fnames,set_title,noise_scale,temp_ratios
    ):
        pl_true_radius = PLANET_PARAMS.radius.to(u.R_earth)
        radius_arr = np.linspace(RADIUS_SCALE_MIN, RADIUS_SCALE_MAX,80)
        red_chi_sq_array = np.zeros((radius_arr.size, log_eps_array.size))
        dist_residual, dist_noise, _s, _coeffs = get_residual_and_noise(
            _epsilon=10**log_epsilon,
            chi_noise_scale=_noise_scale
        )
        dist_residual = bin_image(dist_residual,BIN_WL,BIN_TIME,1)
        dist_noise = bin_image(dist_noise,BIN_WL,BIN_TIME,2)
        binned_wl = bin_image(wl.to_value(u.um),BIN_WL,1,1)[0,:]
        for i, rad in tqdm(enumerate(radius_arr),total=radius_arr.size):
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
                chi_sq_2d = difference**2/dist_noise**2
                long_wl = binned_wl >= CHI2_WL.to_value(u.um)
                chi_sq_2d = chi_sq_2d[:, long_wl]
                chi_sq = np.sum(chi_sq_2d)
                red_chi_sq = chi_sq / (chi_sq_2d.size-2)
                red_chi_sq_array[i, j] = red_chi_sq
        logger.info(f'The lowest value for red chi2 is {np.min(red_chi_sq_array)}')
        with figure_context(figsize=FIGSIZE) as fig:
            ax: plt.Axes = fig.subplots(1, 1)
            im = ax.pcolormesh(
                temp_array, (radius_arr * pl_true_radius).to_value(u.R_earth), (red_chi_sq_array),
                rasterized=True,
                norm=LogNorm(),
                zorder=-100
            )
            ax.set_ylabel('$R_\\mathrm{p}/R_\\oplus$')
            ax.set_xlabel('$T_{\\rm night} / T_{\\rm day}$')
            ax.grid(False)
            fig.colorbar(im, label='$\\chi^2_{\\rm red}$')
            levels = [1, 4, 9, 16, 25,100,225,400]
            def _fmt(x):
                return f'$\\chi^2_{{\\rm red}} = {x:.0f}$'
            im = ax.contour(
                temp_array, (radius_arr * pl_true_radius).to_value(u.R_earth), red_chi_sq_array,
                levels=levels,
                colors='k',
                linestyles='dashed'
            )
            ax.clabel(im, im.levels, inline=True, fontsize=10, fmt=_fmt)
            levels=[1.3,2.1,3.2]
            labels = [ # Zeng+2019 Fig 2
                '$100\\%\\;\\mathrm{Fe}$',
                '$50\\%\\;\\mathrm{H_2O}+50\\%\\;\\mathrm{rock}$',
                '$+2\\%\\;\\mathrm{H_2}$'
            ]
            im=ax.contour(
                temp_array,
                (radius_arr * pl_true_radius).to_value(u.R_earth),
                np.meshgrid(temp_array,(radius_arr * pl_true_radius).to_value(u.R_earth))[1],
                levels=levels,
                colors='w',
                linestyles='-',
                zorder=-99
            )
            def _fmt_atm(x):
                # pylint: disable-next=cell-var-from-loop
                return dict(zip(levels, labels))[x]
            ax.clabel(im,im.levels,inline=True,fontsize=10,fmt=_fmt_atm,zorder=100)
            ax.text(0.5,0.7,'$\\mathrm{Thick\\; H_2\\; Envelope}$',
                    transform=ax.transAxes,fontsize=10,color='w',ha='center',va='center')
            ax.scatter(temp_ratio,pl_true_radius.to_value(u.R_earth),
                       marker='*',c='#c50d15',s=200,edgecolor='w')

            if _title:
                ax.text(0.5,1.05,'GJ 876 d',transform=ax.transAxes,
                        fontsize=16,color='k',ha='center',va='center',fontweight='bold')
            fig.tight_layout()
            fig.savefig(
                paths.figures / f'{PREFIX}_inference_{fname}.pdf')
