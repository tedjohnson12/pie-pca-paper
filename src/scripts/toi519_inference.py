"""
This was previously 4 scripts. We are combining to 1.
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
from common import COLWIDTH, find_eclipse, remove_epoch, figure_context
from toi519_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from toi519_run import (
    get_model, PLANET as PLANET_PARAMS,
    RADIUS_SCALE_MIN, RADIUS_SCALE_MAX,
    TEMP_RATIO_MIN, TEMP_RATIO_MAX,
    SW_MAX, LW_MIN
)


PREFIX = 'toi519'
CACHE_FILE = paths.data / 'toi519-cache.asdf'
IC = 'BIC'
FIGSIZE = (COLWIDTH, 0.7*COLWIDTH)
MAX_BASIS = None
NOISE_SCALE = 1.0
THERMAL_SCALE = 1.0
SEED = 33
FLUX_UNIT = u.Unit('W m-2 um-1')
BIN_WL = 6
BIN_TIME = 4
TEMP_ARRAY = np.linspace(TEMP_RATIO_MIN, TEMP_RATIO_MAX, 150)
RADIUS_ARRAY = np.linspace(RADIUS_SCALE_MIN, RADIUS_SCALE_MAX, 80)
TEMPERATURE_RATIOS = [0.5, 0.99]
HEAT_REDISTRIBUTION = ['mod', 'high']
USE_ECLIPSE = [True, False]
USE_CACHE = [
    [False, False],
    [False, False],
]
CHI2_NOISE_SCALE = [
    np.sqrt([
        16.728182103819513,
        13.789226575837516
    ]),
    np.sqrt([
        13.567046889948333,
        14.488140892357649
    ])
]
LABELS = [
    [
        'b) Eclipse considered',
        'a) Eclipse considered'
    ],
    [
        'd) Eclipse masked',
        'c) Eclipse masked'
    ]
]


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

    def get_residual_and_noise(log_epsilon, chi_noise_scale=1.0, remove_eclipse=True):
        """
        Get the results of a VPIE observation
        """

        _thermal = THERMAL_SCALE*interpolator([log_epsilon])[0, :, :].T
        _data = VSPEC.PhaseAnalyzer.from_model(get_model())
        _wl = _data.wavelength
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
            _total_observed if not remove_eclipse else remove_epoch(
                _total_observed, eclipse_start, eclipse_end),
            _scatter_mag if not remove_eclipse else remove_epoch(
                _scatter_mag, eclipse_start, eclipse_end),
            _cutoff_index,
            True,
            IC,
            max_basis_size=MAX_BASIS
        )
        if remove_eclipse:
            _total_observed = remove_epoch(
                _total_observed, eclipse_start, eclipse_end)
            _uncertainty = remove_epoch(
                _uncertainty, eclipse_start, eclipse_end)
        _residual = _f_rec - _total_observed
        return _residual, _uncertainty, _s, _coeffs, _wl

    log_epsilon_array = temp_to_log_epsilon(TEMP_ARRAY)
    for noise_scale_arr, use_eclipse, use_cache_arr, label_arr in zip(
        CHI2_NOISE_SCALE, USE_ECLIPSE, USE_CACHE, LABELS
    ):
        for temperature_ratio, heat_redistribution, should_use_cache, noise_scale, label in zip(
            TEMPERATURE_RATIOS, HEAT_REDISTRIBUTION, use_cache_arr, noise_scale_arr, label_arr
        ):
            if should_use_cache:
                with asdf.open(CACHE_FILE, mode='r') as f:
                    KEY = f'{use_eclipse}_{heat_redistribution}'
                    red_chi_sq_array = f.tree[KEY][:, :]
            else:
                red_chi_sq_array = np.zeros(
                    (RADIUS_ARRAY.size, log_epsilon_array.size))
                data_epsilon = 10**temp_to_log_epsilon([temperature_ratio])[0]
                data_residual, data_uncertainty, s, coeffs, wl = get_residual_and_noise(
                    np.log10(data_epsilon), chi_noise_scale=noise_scale,
                    remove_eclipse=not use_eclipse
                )
                data_residual = bin_image(data_residual, BIN_WL, BIN_TIME, 1)
                data_noise = bin_image(data_uncertainty, BIN_WL, BIN_TIME, 2)
                binned_wl = bin_image(wl.to_value(u.um), BIN_WL, 1, 1)[0, :]
                for i, rad in tqdm(
                    enumerate(RADIUS_ARRAY), total=RADIUS_ARRAY.size,
                    desc=f'{heat_redistribution} ({use_eclipse})'
                ):
                    for j, log_eps in enumerate(log_epsilon_array):
                        grid_thermal = rad**2*THERMAL_SCALE * \
                            interpolator([log_eps])[0, :, :].T
                        if not use_eclipse:
                            grid_thermal = remove_epoch(
                                grid_thermal, eclipse_start, eclipse_end)
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
                    KEY = f'{use_eclipse}_{heat_redistribution}'
                    af.tree[KEY] = red_chi_sq_array
                    af.write_to(CACHE_FILE)
                    af.close()
                else:
                    KEY = f'{use_eclipse}_{heat_redistribution}'
                    af = asdf.AsdfFile({KEY: red_chi_sq_array})
                    af.write_to(CACHE_FILE)
                    af.close()
                logger.info(
                    f'eclipse: {use_eclipse}, {heat_redistribution}'
                    ' - '
                    f'The lowest value for red chi2 is {np.min(red_chi_sq_array)}'
                )
                with figure_context(figsize=FIGSIZE) as fig:
                    ax: plt.Axes = fig.subplots(1, 1)
                    im = ax.pcolormesh(
                        TEMP_ARRAY, (RADIUS_ARRAY * PLANET_PARAMS.radius).to_value(
                            u.R_jup), (red_chi_sq_array),
                        rasterized=True,
                        norm=LogNorm()
                    )
                    ax.set_ylabel('$R_\\mathrm{p}/R_\\mathrm{J}$')
                    ax.set_xlabel('$T_{\\rm night} / T_{\\rm day}$')
                    ax.grid(False)
                    fig.colorbar(im, label='$\\chi^2_{\\rm red}$')
                    levels = [1, 4, 9, 16, 25, 100, 225, 400]

                    def _fmt(x):
                        return f'$\\chi^2_{{\\rm red}} = {x:.0f}$'
                    im = ax.contour(
                        TEMP_ARRAY, (RADIUS_ARRAY *
                                     PLANET_PARAMS.radius).to_value(u.R_jup), red_chi_sq_array,
                        levels=levels,
                        colors='k',
                        linestyles='dashed'
                    )
                    ax.clabel(im, im.levels, inline=True, fontsize=10, fmt=_fmt)
                    ax.text(0.05, 0.05, label, transform=ax.transAxes, fontsize=10,
                            color='w', ha='left', va='center', fontweight='bold')
                    ax.scatter(temperature_ratio, PLANET_PARAMS.radius.to_value(
                        u.R_jup), marker='*', c='#c50d15', s=200, edgecolor='w')
                    fig.tight_layout()
                    fig.savefig(
                        paths.figures / \
                            f'{PREFIX}_inference_'
                            f'{use_eclipse}_{heat_redistribution}.pdf')
