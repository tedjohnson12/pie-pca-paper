"""
JWST retrieval
"""

from pathlib import Path
from time import time as current_wall_time
import matplotlib.pyplot as plt
import dynesty.plotting
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u
import dynesty
import VSPEC
from scipy.special import erfc
from vpie.retrieve import Parameter, Prior
import vpie
from loguru import logger
import leopy

from jwst_grid import get_interp, LOG_EPSILON_GRID, TRAT_GRID, dt_to_eps
from run_jwst import get_model as get_vspec_model, get_temperature_ratio, TRUE_EPSILON

import paths

PARAMETERS = [
    Parameter(
        name='$T_{\\rm night}/T_{\\rm day}$',
        prior=Prior.uniform(0.11, 0.98),
        truth=get_temperature_ratio(TRUE_EPSILON),
        values=TRAT_GRID
    )
]
NOISE_SCALE = 1
SEED = 10
flux_unit = u.Unit('W m-2 um-1')
OUTFILE_CORNER = paths.figures / 'jwst_corner.pdf'
OUTFILE_LOGL = paths.figures / 'jwst_logl.pdf'
OUTFILE_CHISQ = paths.figures / 'jwst_chisq.pdf'
OUTFILE_RESIDUALS = paths.figures / 'jwst_residuals.pdf'

def get_model(log_epsilon: float, _interp: RegularGridInterpolator):
    return _interp(([log_epsilon],))[0,:,:]

def get_data():
    return VSPEC.PhaseAnalyzer.from_model(get_vspec_model())


if __name__ in '__main__':
    rng = np.random.default_rng(SEED)
    interp = get_interp()
    data = get_data()
    thermal = data.thermal.to_value(flux_unit)
    thermal = interp([0.1])[0]
    noise = data.noise.to_value(flux_unit) * NOISE_SCALE
    total = data.total.to_value(flux_unit)
    total = data.star.to_value(flux_unit) + thermal
    scatter = rng.normal(loc=0,scale=noise)
    print(scatter)
    observed = total + scatter
    wl = data.wavelength.to_value(u.um)
    n_wl = wl.size
    time = data.time.to_value(u.day)
    n_time = time.size
    cutoff_index = np.argwhere(wl > 1.5)[0][0]
    logger.info(f'Cutoff index: {cutoff_index}')
    long_cutoff_index = np.argwhere(wl > 2)[0][0]
    logger.info(f'Long cutoff index: {long_cutoff_index}')
    logger.info(f'Size of wavelength axis: {n_wl}')
    s, coeffs, f_rec = vpie.get_vpie(
        observed.T,
        noise.T,
        cutoff_index=cutoff_index,
        use_mean_error=True,
    )
    data_residual = observed.T - f_rec
    assert data_residual.shape == (n_time, n_wl)
    
    def get_model_residual(log_epsilon:float):
        _thermal = get_model(log_epsilon, interp)
        return _thermal.T - vpie.vpie.get_reconstruction(_thermal.T, coeffs, s)
    
    def get_red_chi_square(log_epsilon:float,noise_scale:float):
        _model_residual = get_model_residual(log_epsilon)
        _res = (data_residual - _model_residual)[:,long_cutoff_index:]
        _noise = noise_scale * noise[long_cutoff_index:,:].T
        chi_sq = np.sum((_res / _noise)**2)
        return chi_sq/_res.size
    
    def get_loglike(log_epsilon:float):
        _model_residual = get_model_residual(log_epsilon)
        _res = (data_residual - _model_residual)[:,long_cutoff_index:]
        _noise = noise[long_cutoff_index:,:].T
        arr_like = erfc(np.abs(_res) / np.sqrt(2)/_noise)
        # return np.sum(np.log(arr_like))/np.sqrt(_res.size)
        # return np.sum(-0.5 * np.log(2*np.pi) - np.log(_noise) - 0.5 * (_res / _noise)**2)
        
    
    def logl(x:np.ndarray):
        return get_loglike(dt_to_eps(x)[0])
    
    # sampler = dynesty.NestedSampler(
    #     logl,
    #     lambda u: np.array([param.prior(_u) for _u, param in zip(u, PARAMETERS)]),
    #     ndim=len(PARAMETERS),
    # )
    # dlogz = 1.0
    # logger.info(f'Dlogz: {dlogz}')
    # sample_start = current_wall_time()
    # sampler.run_nested(dlogz=dlogz)
    # sample_end = current_wall_time()
    # logger.info(f'Wall time: {sample_end - sample_start}')
    
    # results = sampler.results
    # cfig, caxes = dynesty.plotting.cornerplot(
    #             results,
    #             color='#458977',
    #             truth_color='#DE5126',
    #             truths = [p.truth for p in PARAMETERS],
    #             labels = [p.name for p in PARAMETERS],
    #             use_math_text=True,
    #             label_kwargs={'fontsize':16,'rotation':0,'ha':'right'},
    #             hist_kwargs={'alpha':1}
    #         )
    # cfig.subplots_adjust(left=0.18)
    # caxes[-1,-1].set_ylabel('Posterior')
    # caxes[-1,-1].set_xlim(0,1)
    # cfig.savefig(OUTFILE_CORNER)
    plt.close('all')
    y = []
    x = np.linspace(0.11, 0.95, 100)
    for _x in x:
        y.append(logl([_x]))
    plt.plot(x, y)
    plt.axvline(PARAMETERS[0].truth)
    plt.xlabel('$T_{\\mathrm{night}}/T_{\\mathrm{day}}$')
    plt.ylabel('$\\log\\mathcal{L}$')
    plt.savefig(OUTFILE_LOGL)
    plt.close('all')
    
    x = np.linspace(0.11, 0.95, 1000)
    x = [0.1]
    distance = [10]
    
    for d in distance:
        scale = (d/10)**2
        y = []
        for _x in x:
            y.append(get_red_chi_square([_x],scale))
        plt.plot(x, y, label=f'{d} pc')
    plt.axvline(PARAMETERS[0].truth)
    # plt.yscale('log')
    # plt.yticks(np.arange(2,26))
    plt.xlabel('$T_{\\mathrm{night}}/T_{\\mathrm{day}}$')
    plt.ylabel('$\\chi^2_{\\mathrm{red}}$')
    # plt.yscale('log')
    # plt.ylabel('distance (pc)')
    plt.legend()
    plt.savefig(OUTFILE_CHISQ)
    plt.close('all')
    
    
    
    fig = plt.figure(figsize=(20,8))
    axes = fig.subplots(LOG_EPSILON_GRID.size+1,2)
    
    for i,loge in enumerate(LOG_EPSILON_GRID):
        lax = axes[i,0]
        rax = axes[i,1]
        _model_residual = get_model_residual(loge)
        _res = (data_residual - _model_residual)[:,long_cutoff_index:]
        lax.imshow(_model_residual, cmap='bwr',vmin=-1e-17,vmax=1e-17)
        rax.imshow(_res)
        # lax.set_title(f'log epsilon = {loge}')
        # rax.set_title(f'residuals')
        # lax.set_xlabel('wavelength')
        # rax.set_xlabel('wavelength')
        # lax.set_ylabel('time')
        # rax.set_ylabel('time')
    lax = axes[-1,0]
    lax.imshow(data_residual/observed.T, cmap='bwr',vmin=-1e-19,vmax=1e-19)
    plt.savefig(OUTFILE_RESIDUALS)
    plt.close('all')