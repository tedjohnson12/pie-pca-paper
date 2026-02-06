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

from vpie import vpie
import VSPEC

import paths
from common import bin_image, figure_context, COLWIDTH, fold_image
from mirecle2d_grid import get_interp as get_interp_200cm, dt_to_eps as temp_to_log_epsilon
from mirecle50cm_grid import get_interp as get_interp_50cm
from mirecle564cm_grid import get_interp as get_interp_564cm
from mirecle2m_run import get_model as get_model_200cm, PLANET as PLANET_PARAMS_200CM
from mirecle50cm_run import get_model as get_model_50cm, PLANET as PLANET_PARAMS_50CM
from mirecle564cm_run import get_model as get_model_564cm, PLANET as PLANET_PARAMS_564CM

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

APERTURES = [50,200,564]
INTERPOLATORS = [get_interp_50cm, get_interp_200cm, get_interp_564cm]
MODEL_GETTERS = [get_model_50cm, get_model_200cm, get_model_564cm]
PLANET_PARAMS = [PLANET_PARAMS_50CM, PLANET_PARAMS_200CM, PLANET_PARAMS_564CM]
SHOULD_FOLD = [True, False, False]
TITLES = ['0.5 m mirror', '2 m mirror', 'JWST-like mirror']
FOLD = 67

TEMPERATURE_RATIOS = [0.05,0.5,0.99]
LABELS = ['full','half','null']
USE_CACHE = (
    [True, True, True],
    [True, True, True],
    [True, True, True]
)
CHI2_NOISE_SCALE = (
        np.sqrt([
        # 82.8597124847814,
        # 13.080702255890222,
        # 82.23186373659043
        10.638529588086413,
        3.2247468815225324,
        10.605978290205305
    ]),
        np.sqrt([
        2.9344707963359737,
        2.9571702536063524,
        2.946854993074575
    ]),
        np.sqrt([
        3.027211584699654,
        2.986337237458304,
        2.947257712176422
    ])
)

def get_residual_and_noise(chi_noise_scale, epsilon, interpolator, get_model,should_fold=False):
    _thermal = THERMAL_SCALE*interpolator([np.log10(epsilon)])[0, :, :].T
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
    if should_fold:
        _total_observed = fold_image(_total_observed, FOLD,1)
        _scatter_mag = fold_image(_scatter_mag, FOLD,2)
        _uncertainty = fold_image(_uncertainty, FOLD,2)
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

    temp_array = np.linspace(0.05, 0.99, 150)
    log_eps_array = (temp_to_log_epsilon(temp_array))
    
    
    for aperture, noise_scales, interpolator_initializer, get_model, planet_params, use_cache_arr, should_fold, title in zip(
        APERTURES, CHI2_NOISE_SCALE, INTERPOLATORS, MODEL_GETTERS, PLANET_PARAMS, USE_CACHE, SHOULD_FOLD, TITLES
    ):
        interpolator = interpolator_initializer()
        pl_true_radius = planet_params.radius.to(u.R_earth)
        for temperature_ratio, label, noise_scale, should_use_cache in zip(TEMPERATURE_RATIOS, LABELS, noise_scales, use_cache_arr):
            radius_arr = np.linspace(0.05, 3.2,80)
            if should_use_cache:
                with asdf.open(CACHE_FILE,mode='r') as f:
                    key = f'{aperture}_{label}'
                    red_chi_sq_array = f.tree[key][:,:]
            else: 
                red_chi_sq_array = np.zeros((radius_arr.size, log_eps_array.size))
                epsilon = 10**temp_to_log_epsilon([temperature_ratio])[0]
                data_residual, data_noise, _s, _coeffs, _wl = get_residual_and_noise(
                    epsilon=epsilon,
                    chi_noise_scale=noise_scale,
                    interpolator=interpolator,
                    get_model=get_model
                )
                # if should_fold:
                #     data_residual = fold_image(data_residual,FOLD,1)
                #     data_noise = fold_image(data_noise,FOLD,2)
                data_residual = bin_image(data_residual,BIN_WL,BIN_TIME,1)
                data_noise = bin_image(data_noise,BIN_WL,BIN_TIME,2)
                binned_wl = bin_image(_wl.to_value(u.um),BIN_WL,1,1)[0,:]
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
                        # if should_fold:
                        #     grid_residual = fold_image(grid_residual,FOLD,1)
                        binned_grid_residual = bin_image(grid_residual, BIN_WL,BIN_TIME, 1)
                        difference = binned_grid_residual - data_residual
                        chi_sq_spec = difference**2/(data_noise )**2
                        long_wl = binned_wl >= CHI2_WL.to_value(u.um)
                        chi_sq_spec = chi_sq_spec[:, long_wl]
                        chi_sq = np.sum(chi_sq_spec)
                        red_chi_sq = chi_sq / (chi_sq_spec.size+2)
                        red_chi_sq_array[i, j] = red_chi_sq
                if CACHE_FILE.exists():
                    af = asdf.open(CACHE_FILE,mode='rw')
                    key = f'{aperture}_{label}'
                    af.tree[key] = red_chi_sq_array
                    af.write_to(CACHE_FILE)
                    af.close()
                else:
                    key = f'{aperture}_{label}'
                    af = asdf.AsdfFile({key: red_chi_sq_array})
                    af.write_to(CACHE_FILE)
                    af.close()
                logger.info(f'{aperture:.1f} m, {label} - The lowest value for red chi2 is {np.min(red_chi_sq_array)}')
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
                levels = [1, 4, 9, 16, 25, 100]
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
                ax.scatter(temperature_ratio,pl_true_radius.to_value(u.R_earth),marker='*',c='#c50d15',s=200,edgecolor='w')
                if label == 'null':
                    ax.text(0.5,1.05,title,transform=ax.transAxes,fontsize=16,color='k',ha='center',va='center',fontweight='bold')
                fig.tight_layout()
                fig.savefig(
                    paths.figures / f'{PREFIX}_{aperture}cm__{label}.pdf',
                    bbox_inches='tight',)
