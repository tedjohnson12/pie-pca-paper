"""
This was previously 3 scripts. We are combining to 1.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy import units as u
from loguru import logger
from tqdm.auto import tqdm
import asdf

from vpie import vpie
import VSPEC
from vpie import bin_image

import paths
from common import COLWIDTH, find_eclipse, figure_context
from proxb_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from proxb_run import (
    get_model, PLANET as PLANET_PARAMS,
    RADIUS_SCALE_MIN, RADIUS_SCALE_MAX,
    TEMP_RATIO_MIN, TEMP_RATIO_MAX,
    SW_MAX, LW_MIN
)

PREFIX = 'proxb'
CACHE_FILE = paths.data / 'proxb-cache.asdf'
IC = 'BIC'
FIGSIZE = (COLWIDTH, 0.7*COLWIDTH)
MAX_BASIS = None
NOISE_SCALE = 1.0
THERMAL_SCALE = 1.0
SEED = 33
FLUX_UNIT = u.Unit('W m-2 um-1')
BIN_WL = 6
BIN_TIME = 3

TEMP_ARRAY = np.linspace(TEMP_RATIO_MIN, TEMP_RATIO_MAX, 150)
RADIUS_ARRAY = np.linspace(RADIUS_SCALE_MIN, RADIUS_SCALE_MAX, 80)
TEMPERATURE_RATIOS = [0.05, 0.5, 0.99]
HEAT_REDISTRIBUTION = ['none', 'mod', 'high']
SET_TITLE = [False, False, True]
USE_CACHE = [
    False, False, False
]
CHI2_NOISE_SCALE = np.sqrt([
    4.225832464514367,
    5.076399774891293,
    5.2804814260398185
])

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

    def get_residual_and_noise(chi_noise_scale, epsilon):
        """
        Get the results of a VPIE observation
        """
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
        _cutoff_index = np.argwhere(wl > SW_MAX)[0][0]
        _s, _coeffs, _f_rec = vpie.get_vpie(
            _total_observed,
            _scatter_mag,
            _cutoff_index,
            True,
            IC,
            max_basis_size=MAX_BASIS
        )
        _residual = _f_rec - _total_observed
        return _residual, _uncertainty, _s, _coeffs, _data.wavelength

    log_epsilon_array = temp_to_log_epsilon(TEMP_ARRAY)
    for noise_scale, should_use_cache, temperature_ratio, \
            heat_redistribution, set_title in zip(
                CHI2_NOISE_SCALE, USE_CACHE, TEMPERATURE_RATIOS, HEAT_REDISTRIBUTION,
                SET_TITLE
            ):
        if should_use_cache:
            with asdf.open(CACHE_FILE, mode='r') as f:
                KEY = heat_redistribution
                red_chi_sq_array = f.tree[KEY][:, :]
        else:
            red_chi_sq_array = np.zeros(
                (RADIUS_ARRAY.size, log_epsilon_array.size))
            data_epsilon = 10**temp_to_log_epsilon([temperature_ratio])[0]
            data_residual, data_uncertainty, s, coeffs, wl = get_residual_and_noise(
                epsilon=data_epsilon, chi_noise_scale=noise_scale
            )
            data_residual = bin_image(data_residual, BIN_WL, BIN_TIME, 1)
            data_noise = bin_image(data_uncertainty, BIN_WL, BIN_TIME, 2)
            binned_wl = bin_image(wl.to_value(u.um), BIN_WL, 1, 1)[0, :]
            for i, rad in tqdm(
                enumerate(RADIUS_ARRAY), total=RADIUS_ARRAY.size,
                desc=heat_redistribution
            ):
                for j, log_eps in enumerate(log_epsilon_array):
                    grid_thermal = rad**2*THERMAL_SCALE * \
                        interpolator([log_eps])[0, :, :].T
                    grid_reconstruction = vpie.get_reconstruction(
                        grid_thermal,
                        coeffs,
                        s
                    )
                    grid_residual = grid_reconstruction - grid_thermal
                    binned_grid_residual = bin_image(
                        grid_residual, BIN_WL, BIN_TIME, 1)
                    difference = binned_grid_residual - data_residual
                    chi_sq_2d = difference**2/(data_noise)**2
                    long_wl = binned_wl >= LW_MIN.to_value(u.um)
                    chi_sq_2d = chi_sq_2d[:, long_wl]
                    chi_sq = np.sum(chi_sq_2d)
                    red_chi_sq = chi_sq / (chi_sq_2d.size-2)
                    red_chi_sq_array[i, j] = red_chi_sq
            if CACHE_FILE.exists():
                af = asdf.open(CACHE_FILE, mode='rw')
                KEY = heat_redistribution
                af.tree[KEY] = red_chi_sq_array
                af.write_to(CACHE_FILE)
                af.close()
            else:
                KEY = heat_redistribution
                af = asdf.AsdfFile({KEY: red_chi_sq_array})
                af.write_to(CACHE_FILE)
                af.close()
            logger.info(
                heat_redistribution +
                ' - '
                f'The lowest value for red chi2 is {np.min(red_chi_sq_array)}'
            )
            with figure_context(figsize=FIGSIZE) as fig:
                ax: plt.Axes = fig.subplots(1, 1)
                im = ax.pcolormesh(
                    TEMP_ARRAY, (RADIUS_ARRAY * PLANET_PARAMS.radius).to_value(
                        u.R_earth), (red_chi_sq_array),
                    rasterized=True,
                    norm=LogNorm(),
                    zorder=-100
                )
                ax.set_ylabel('$R_\\mathrm{p}/R_\\oplus$')
                ax.set_xlabel('$T_{\\rm night} / T_{\\rm day}$')
                ax.grid(False)
                fig.colorbar(im, label='$\\chi^2_{\\rm red}$')
                levels = [1, 4, 9, 16, 25, 100, 225, 400]

                def _fmt(x):
                    return f'$\\chi^2_{{\\rm red}} = {x:.0f}$'
                im = ax.contour(
                    TEMP_ARRAY, (RADIUS_ARRAY *
                                 PLANET_PARAMS.radius).to_value(u.R_earth), red_chi_sq_array,
                    levels=levels,
                    colors='k',
                    linestyles='dashed'
                )
                ax.clabel(im, im.levels, inline=True, fontsize=10, fmt=_fmt)
                levels = [0.75, 1, 1.3, 2.6]
                labels = [  # See Zeng+2019
                    # Also note that Lopez & Fortney (2014) show that
                    # insolation is not important in determining the radius
                    '$100\\%\\;\\mathrm{Fe}$',
                    'Earth-like',
                    '50% rock & 50% H$_2$O',
                    '+2% H$_2$',
                ]
                im = ax.contour(
                    TEMP_ARRAY, (RADIUS_ARRAY *
                                 PLANET_PARAMS.radius).to_value(u.R_earth),
                    np.meshgrid(
                        TEMP_ARRAY, (RADIUS_ARRAY *
                                     PLANET_PARAMS.radius).to_value(u.R_earth)
                    )[1],
                    levels=levels,
                    colors='w',
                    linestyles='-',
                    zorder=-99
                )

                def _fmt_atm(x):
                    # pylint: disable-next=cell-var-from-loop
                    return dict(zip(levels, labels))[x]
                ax.clabel(im, im.levels, inline=True,
                          fontsize=10, fmt=_fmt_atm)
                # ax.text(0.05, 0.05, label, transform=ax.transAxes, fontsize=10,
                #         color='w', ha='left', va='center', fontweight='bold')
                ax.scatter(temperature_ratio, PLANET_PARAMS.radius.to_value(
                    u.R_earth), marker='*', c='#c50d15', s=200, edgecolor='w')
                if set_title:
                    ax.text(0.5,1.05,'Proxima Centauri b',transform=ax.transAxes,
                        fontsize=16,color='k',ha='center',va='center',fontweight='bold')
                fig.tight_layout()
                fig.savefig(
                    paths.figures /
                    f'{PREFIX}_inference_'
                    f'{heat_redistribution}.pdf')
