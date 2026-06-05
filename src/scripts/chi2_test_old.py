import contextlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy import units as u
from loguru import logger

from vpie import vpie
import VSPEC

import paths
from common import bin_image, COLWIDTH

from gj876_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from gj876_run import get_model, PLANET as PLANET_PARAMS

PREFIX = 'chi2_test'
IC = 'BIC'
FIGSIZE = (COLWIDTH, 1*COLWIDTH)
MAX_BASIS = None
FLUX_UNIT = u.Unit('W m-2 um-1')
NOISE_SCALE = 1.0
CHI2_NOISE_SCALE = np.sqrt(1.0)
THERMAL_SCALE = 1.0
SEED = 11
CUTOFF_WL = 0.8*u.um
CHI2_WL = 4.0*u.um
BIN_WL = 6
BIN_TIME = 3
LEGEND_TEXT_SIZE = 8
AXIS_TEXT_SIZE = 10

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
    
    def get_residual_and_noise(chi_noise_scale, epsilon):
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
        return _residual, _uncertainty, _s, _coeffs, _thermal
    
    residual_baseline, uncertainty_baseline, s, coeffs, thermal = get_residual_and_noise(
            CHI2_NOISE_SCALE, 0.1)
    binned_residual_baseline = bin_image(residual_baseline,BIN_WL,BIN_TIME,1)
    binned_thermal_baseline = bin_image(thermal,BIN_WL,BIN_TIME,1)
    binned_noise_baseline = bin_image(uncertainty_baseline,BIN_WL,BIN_TIME,2)
    binned_wl = bin_image(wl.to_value(u.um),BIN_WL,1,1)[0,:]
    binned_time = bin_image(time.to_value(u.day),BIN_TIME,1,1)[0,:]
    
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(binned_time,binned_thermal_baseline[:,-1],label='No heat redistribution (baseline)')
    temps = [0.5,0.99]
    labels = [
        'Moderate heat redistribution',
        'Full heat redistribution',
    ]
    
    for temp, label in zip(temps, labels):
        log_epsilon = temp_to_log_epsilon([temp])[0]
        residual, uncertainty, s, coeffs, thermal = get_residual_and_noise(
            CHI2_NOISE_SCALE, 10**log_epsilon)
        binned_residual = bin_image(residual,BIN_WL,BIN_TIME,1)
        binned_thermal = bin_image(thermal,BIN_WL,BIN_TIME,1)
        binned_noise = bin_image(uncertainty,BIN_WL,BIN_TIME,2)
        chi2_spec = (binned_residual-binned_residual_baseline)**2/binned_noise_baseline**2
        chi2_lw = chi2_spec[:, binned_wl > CHI2_WL.to_value(u.um)]
        red_chi2 = np.sum(chi2_lw) / (chi2_lw.size-2)
        logger.info(f'{temp:.2f}: {red_chi2:.2f}')
        _label = f'{label} ($\\chi^2_{{\\rm red}}={red_chi2:.2f}$)'
        ax.plot(binned_time,binned_thermal[:,-1],label=_label)
    ax.set_xlabel('Time (days)', fontsize=AXIS_TEXT_SIZE, fontfamily='serif')
    ax.set_ylabel('Residual @ LW ($\\mathrm{W m^{-2} \\mu m^{-1}}$)', fontsize=AXIS_TEXT_SIZE, fontfamily='serif')
    ax.plot(binned_time,binned_noise_baseline[:,-1],color='k',ls='--',label='Data uncertainty')
    ax.legend(prop={'size': LEGEND_TEXT_SIZE, 'family': 'serif'})
    ax.set_facecolor('w')
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0],ylims[1]*1.3)
    ax.set_title('')
    fig.tight_layout()
    fig.savefig(paths.figures / 'chi2_test.pdf')