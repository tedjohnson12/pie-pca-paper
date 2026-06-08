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
import asdf
from time import time

from vpie import vpie
import VSPEC
from vpie import bin_image, fold_image as fold_image

import paths
from common import figure_context, COLWIDTH
from mirecle2d_grid import get_interp as get_interp_200cm, dt_to_eps as temp_to_log_epsilon
from mirecle50cm_grid import get_interp as get_interp_50cm
from mirecle564cm_grid import get_interp as get_interp_564cm
from mirecle2m_run import get_model as get_model_200cm, PLANET as PLANET_PARAMS_200CM
from mirecle50cm_run import get_model as get_model_50cm, PLANET as PLANET_PARAMS_50CM
from mirecle564cm_run import get_model as get_model_564cm, PLANET as PLANET_PARAMS_564CM

from proxb_run import RADIUS_SCALE_MIN, RADIUS_SCALE_MAX, TEMP_RATIO_MIN, TEMP_RATIO_MAX


PREFIX = 'mirecle2d'
IC = 'BIC'
MAX_BASIS = None
NOISE_SCALE = 1.0
THERMAL_SCALE = 1.0
SEED = 33
FLUX_UNIT = u.Unit('W m-2 um-1')
CUTOFF_WL = 3*u.um
CHI2_WL = 8.0*u.um
BIN_WL = 6
BIN_TIME = 3
FIGSIZE = (COLWIDTH, 0.7*COLWIDTH)
CACHE_FILE = paths.data / 'mirecle-cache.asdf'

APERTURES = [50, 200, 564]
INTERPOLATORS = [get_interp_50cm, get_interp_200cm, get_interp_564cm]
MODEL_GETTERS = [get_model_50cm, get_model_200cm, get_model_564cm]
PLANET_PARAMS = [PLANET_PARAMS_50CM, PLANET_PARAMS_200CM, PLANET_PARAMS_564CM]
SHOULD_FOLD = [True, False, False]
TITLES = ['0.5 m mirror', '2 m mirror', 'JWST-like mirror']
FOLD = 67

TEMPERATURE_RATIOS = [0.05, 0.5, 0.99]
LABELS = ['none', 'mod', 'high']
USE_CACHE = (
    [False, False, False],
    [False, False, False],
    [False, False, False],
)
CHI2_NOISE_SCALE = (
    np.sqrt([
            1.0655550464948622,
            1.1343578145456072,
            1.2580238734025708
            ]),
    np.sqrt([
            3.0076883373713144,
            3.0311426843080365,
            3.0205032689665336
            ]),
    np.sqrt([
            3.103367850981406,
            3.061465218274865,
            3.0214025602814893
            ])
)


def _get_residual_and_noise(
    chi_noise_scale, _epsilon, _interpolator, _get_model, _should_fold=False
):
    _thermal = THERMAL_SCALE*_interpolator([np.log10(_epsilon)])[0, :, :].T
    _data = VSPEC.PhaseAnalyzer.from_model(_get_model())
    _wl = _data.wavelength
    _rng = np.random.default_rng(SEED)
    _stellar = _data.star.T.to_value(FLUX_UNIT)
    _scatter_mag = _data.noise.T.to_value(
        FLUX_UNIT) * NOISE_SCALE
    _total_true = _stellar + _thermal
    _scatter = _rng.normal(loc=0, scale=_scatter_mag)
    _uncertainty = _scatter_mag * chi_noise_scale
    _total_observed = _total_true + _scatter
    if _should_fold:
        _total_observed = fold_image(_total_observed, FOLD, 1)
        _scatter_mag = fold_image(_scatter_mag, FOLD, 2)
        _uncertainty = fold_image(_uncertainty, FOLD, 2)
        _total_observed = bin_image(_total_observed, 1, 4, 1)
        _scatter_mag = bin_image(_scatter_mag, 1, 4, 1)
        _uncertainty = bin_image(_uncertainty, 1, 4, 1)
    _cutoff_index = np.argwhere(_wl > CUTOFF_WL)[0][0]
    _s, _coeffs, _f_rec = vpie.get_vpie(
        _total_observed,
        _scatter_mag,
        _cutoff_index,
        True,
        IC,
        max_basis_size=MAX_BASIS
    )
    _residual = _f_rec - _total_observed
    return _residual, _uncertainty, _s, _coeffs, _wl


if __name__ in '__main__':
    plt.style.use('bmh')

    temp_array = np.linspace(TEMP_RATIO_MIN, TEMP_RATIO_MAX, 150)
    log_eps_array = temp_to_log_epsilon(temp_array)

    for aperture, noise_scales, interpolator_initializer, get_model, \
            planet_params, use_cache_arr, should_fold, title in zip(
                APERTURES, CHI2_NOISE_SCALE, INTERPOLATORS, MODEL_GETTERS,
                PLANET_PARAMS, USE_CACHE, SHOULD_FOLD, TITLES
            ):
        if all(use_cache_arr):
            INTERPOLATOR = None
        else:
            init_start = time()
            INTERPOLATOR = interpolator_initializer()
            init_end = time()
            logger.info(
                f'Interpolator took {init_end-init_start:.2f} seconds to load.')
        pl_true_radius = planet_params.radius.to(u.R_earth)
        for temperature_ratio, label, noise_scale, should_use_cache in zip(
            TEMPERATURE_RATIOS, LABELS, noise_scales, use_cache_arr
        ):
            radius_arr = np.linspace(RADIUS_SCALE_MIN, RADIUS_SCALE_MAX, 80)
            if should_use_cache:
                with asdf.open(CACHE_FILE, mode='r') as f:
                    KEY = f'{aperture}_{label}'
                    red_chi_sq_array = f.tree[KEY][:, :]
            else:
                red_chi_sq_array = np.zeros(
                    (radius_arr.size, log_eps_array.size))
                epsilon = 10**temp_to_log_epsilon([temperature_ratio])[0]
                logger.info('About to get residual and noise.')
                res_start = time()
                data_residual, data_noise, _s, _coeffs, _wl = _get_residual_and_noise(
                    _epsilon=epsilon,
                    chi_noise_scale=noise_scale,
                    _interpolator=INTERPOLATOR,
                    _get_model=get_model,
                    _should_fold=should_fold
                )
                res_end = time()
                logger.info(
                    f'Residual and noise took {res_end-res_start:.2f} seconds.')
                logger.info('About to bin images.')
                data_residual = bin_image(data_residual, BIN_WL, BIN_TIME, 1)
                data_noise = bin_image(data_noise, BIN_WL, BIN_TIME, 2)
                binned_wl = bin_image(_wl.to_value(u.um), BIN_WL, 1, 1)[0, :]
                for i, rad in tqdm(enumerate(radius_arr), total=radius_arr.size):
                    for j, log_eps in enumerate(log_eps_array):
                        grid_thermal = rad**2*THERMAL_SCALE * \
                            INTERPOLATOR([log_eps])[0, :, :].T
                        if should_fold:
                            grid_thermal = fold_image(grid_thermal, FOLD, 1)
                            grid_thermal = bin_image(grid_thermal, 1, 4, 1)
                        grid_reconstruction = vpie.get_reconstruction(
                            grid_thermal,
                            _coeffs,
                            _s
                        )
                        grid_residual = grid_reconstruction - grid_thermal
                        binned_grid_residual = bin_image(
                            grid_residual, BIN_WL, BIN_TIME, 1)
                        difference = binned_grid_residual - data_residual
                        chi_sq_spec = difference**2/(data_noise)**2
                        long_wl = binned_wl >= CHI2_WL.to_value(u.um)
                        chi_sq_spec = chi_sq_spec[:, long_wl]
                        chi_sq = np.sum(chi_sq_spec)
                        red_chi_sq = chi_sq / (chi_sq_spec.size-2)
                        red_chi_sq_array[i, j] = red_chi_sq
                if CACHE_FILE.exists():
                    af = asdf.open(CACHE_FILE, mode='rw')
                    KEY = f'{aperture}_{label}'
                    af.tree[KEY] = red_chi_sq_array
                    af.write_to(CACHE_FILE)
                    af.close()
                else:
                    KEY = f'{aperture}_{label}'
                    af = asdf.AsdfFile({KEY: red_chi_sq_array})
                    af.write_to(CACHE_FILE)
                    af.close()
                logger.info(f'{aperture:.1f} cm, {label} - '
                            f'The lowest value for red chi2 is {np.min(red_chi_sq_array)}')
            with figure_context(figsize=FIGSIZE) as fig:
                ax: plt.Axes = fig.subplots(1, 1)
                im = ax.pcolormesh(
                    temp_array,
                    (radius_arr * pl_true_radius).to_value(u.R_earth),
                    (red_chi_sq_array),
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
                    temp_array, (radius_arr *
                                 pl_true_radius).to_value(u.R_earth), red_chi_sq_array,
                    levels=levels,
                    colors='k',
                    linestyles='dashed'
                )
                ax.clabel(im, im.levels, inline=True, fontsize=10, fmt=_fmt)
                levels = [0.75, 1, 1.3, 2.6]
                labels = [  # See Zeng+2019
                    # Also note that Lopez & Fortney (2014)
                    # show that insolation is not important in determining the radius
                    '$100\\%\\;\\mathrm{Fe}$',
                    'Earth-like',
                    '50% rock & 50% H$_2$O',
                    '+2% H$_2$',
                    # '$\\mathrm{Thick\\; H_2\\; Envelope}$'
                ]
                im = ax.contour(
                    temp_array,
                    (radius_arr * pl_true_radius).to_value(u.R_earth),
                    np.meshgrid(temp_array, (radius_arr *
                                pl_true_radius).to_value(u.R_earth))[1],
                    levels=levels,
                    colors='w',
                    linestyles='-',
                    zorder=-99
                )

                def _fmt_atm(x):
                    # pylint: disable-next=cell-var-from-loop
                    return dict(zip(levels, labels))[x]
                ax.clabel(im, im.levels, inline=True,
                          fontsize=10, fmt=_fmt_atm, zorder=100)
                ax.scatter(temperature_ratio, pl_true_radius.to_value(u.R_earth),
                           marker='*', c='#c50d15', s=200, edgecolor='w')
                if label == 'null':
                    ax.text(0.5, 1.05, title, transform=ax.transAxes,
                            fontsize=16, color='k', ha='center', va='center', fontweight='bold')
                fig.tight_layout()
                fig.savefig(
                    paths.figures / f'{PREFIX}_{aperture}cm__{label}.pdf',
                    bbox_inches='tight',)
