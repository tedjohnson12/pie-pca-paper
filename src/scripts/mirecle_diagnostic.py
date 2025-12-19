import contextlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy import units as u
from loguru import logger
from scipy.optimize import minimize_scalar

from vpie import vpie
import VSPEC

import paths
from mirecle_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from mirecle_run import get_model, PLANET as PLANET_PARAMS

PREFIX = 'mirecle'
IC = 'AIC'
MAX_BASIS = 4
TRUE_TEMPERATURE_RATIO = 0.1
TRUE_LOG_EPSILON = temp_to_log_epsilon([TRUE_TEMPERATURE_RATIO])
NOISE_SCALE = 1.0/np.sqrt(2)
CHI2_NOISE_SCALE = 1
THERMAL_SCALE = 1.0
SEED = 11
FLUX_UNIT = u.Unit('W m-2 um-1')
CUTOFF_WL = 5*u.um
CHI2_WL = 10*u.um
TIME_UNIT = u.day
WL_UNIT = u.um
INTERACTIVE = True


@contextlib.contextmanager
def figure_context(*args, **kwargs):
    fig: plt.Figure = plt.figure(*args, **kwargs)
    yield fig
    if INTERACTIVE:
        plt.show()
    plt.close(fig)

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
    true_uncertainty = data.noise.T.to_value(FLUX_UNIT)
    uncertainty = true_uncertainty * NOISE_SCALE
    total_true = stellar + thermal
    scatter = rng.normal(loc=0, scale=uncertainty)
    total_observed = total_true + scatter
    null_observed = stellar + scatter
    
    with figure_context(figsize=(6, 12)) as fig:
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        axes = [ax1, ax2, ax3]
        denominators = [
            1, stellar/1e6, uncertainty
        ]
        labels = [
            r'$\mathrm{W m^{-2} \mu m^{-1}}$',
            'Star contrast (ppm)',
            'Noise contrast'
        ]
        for ax, denom, label in zip(axes, denominators, labels):
            im1 = ax.pcolormesh(wl.to_value(WL_UNIT), time.to_value(TIME_UNIT), (thermal/denom), rasterized=True, cmap='afmhot_r')
            ax.set_xlabel('Wavelength ($\\rm \\mu m$)')
            ax.set_ylabel('Time (days)')
            ax.axvline(CUTOFF_WL.to_value(WL_UNIT), ls='--', c='k')
            ax.axvline(CHI2_WL.to_value(WL_UNIT), ls='--', c='k')
            cbar1 = fig.colorbar(im1, ax=ax, orientation='vertical', shrink=0.8,label=label)
    
    cutoff_index = np.argwhere(wl > CUTOFF_WL)[0][0]
    long_cutoff_index = np.argwhere(wl > CHI2_WL)[0][0]
    logger.info(
        f'For a short-wave cutoff of {CUTOFF_WL}, we choose a cutoff index of {cutoff_index}. Total wl axis size is {wl.size}')
    logger.info(
        f'For a long-wave cutoff of {CHI2_WL}, we choose a cutoff index of {long_cutoff_index}. Total wl axis size is {wl.size}')
    s, coeffs, f_rec = vpie.get_vpie(
        total_observed,
        uncertainty,
        cutoff_index,
        True,
        IC
    )
    s_null, coeffs_null, f_rec_null = vpie.get_vpie(
        null_observed,
        uncertainty,
        cutoff_index,
        True,
        IC
    )
    
    
    with figure_context(figsize=(6, 4)) as fig:
        ax = fig.add_subplot(1,1,1)
        for i, _s in enumerate(s):
            k = coeffs[:,i]
            ax.plot(time.to_value(TIME_UNIT), k, label=f'{_s}')
        for i,_s in enumerate(s_null):
            k = coeffs_null[:,i]
            ax.plot(time.to_value(TIME_UNIT), k, label=f'{_s}',ls='--')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{PREFIX} {IC}')
        ax.legend()
    

    residual = f_rec - total_observed
    null_residual = f_rec_null - null_observed
    
    with figure_context(figsize=(12, 12)) as fig:
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3,2,4)
        ax5 = fig.add_subplot(3,2,5)
        ax6 = fig.add_subplot(3,2,6)
        axes = [
            [ax1, ax3, ax5],
            [ax2, ax4, ax6]
        ]
        numerators = [residual, null_residual]
        denominators = [
            1e-16, stellar/1e6, uncertainty
        ]
        labels = [
            r'$10^{-16}\mathrm{W m^{-2} \mu m^{-1}}$',
            'Star contrast (ppm)',
            'Noise contrast'
        ]
        ax1.set_title('Thermal Contrast')
        ax2.set_title('Null Contrast')
        for _axes, numerator in zip(axes, numerators):
            for ax, denom, label in zip(_axes, denominators, labels):
                Z = numerator/denom
                vminmax = np.max(np.abs(Z))
                im1 = ax.pcolormesh(wl.to_value(WL_UNIT), time.to_value(TIME_UNIT), Z, rasterized=True, cmap='bwr',vmin=-vminmax,vmax=vminmax)
                ax.set_xlabel('Wavelength ($\\rm \\mu m$)')
                ax.set_ylabel('Time (days)')
                ax.axvline(CUTOFF_WL.to_value(WL_UNIT), ls='--', c='k')
                ax.axvline(CHI2_WL.to_value(WL_UNIT), ls='--', c='k')
                regions = [wl < CUTOFF_WL, (wl>=CUTOFF_WL) & (wl<CHI2_WL), wl>=CHI2_WL]
                for reg in regions:
                    Zreg = Z[:,reg]
                    mean = np.mean(Zreg)
                    stdev = np.std(Zreg)
                    rms = np.sqrt(np.mean(Zreg**2))
                    ma = np.mean(np.abs(Zreg))
                    s = f'Mean: {mean:.1f}\n' \
                      + f'Std:  {stdev:.1f}\n' \
                      + f'rms:  {rms:.1f}\n' \
                      + f'mae:  {ma:.1f}'
                    ax.text(wl[reg][0].to_value(WL_UNIT),6,s)
                    
                cbar1 = fig.colorbar(im1, ax=ax, orientation='vertical', shrink=0.8,label=label)
        
    def bin_image(im: np.ndarray, nwl:int, ntime: int, power: int):
        def add(*args):
            _sum = args[0] * 0
            for arg in args:
                _sum += arg**power
            return _sum**(1/power) / len(args)
        im = np.atleast_2d(im)
        original_size_time, original_size_wl = im.shape
        new_size_time = int(np.ceil(original_size_time/ntime))
        new_size_wl = int(np.ceil(original_size_wl/nwl))
        out_arr = np.zeros((new_size_time, new_size_wl))
        for i in range(new_size_time):
            for j in range(new_size_wl):
                sub_arr = (im[i*ntime:min((i+1)*ntime,original_size_time), j*nwl:min((j+1)*nwl,original_size_wl)]).flatten()
                val = add(*sub_arr)
                out_arr[i,j] = val
        return out_arr
    
    BIN_TIME = 4
    BIN_WL = 6
    with figure_context(figsize=(12, 12)) as fig:
        fig.text(0.5,0.98,f'WL BIN = {BIN_WL}, TIME BIN = {BIN_TIME}')
        ax1 = fig.add_subplot(3, 2, 1)
        ax2 = fig.add_subplot(3, 2, 2)
        ax3 = fig.add_subplot(3, 2, 3)
        ax4 = fig.add_subplot(3,2,4)
        ax5 = fig.add_subplot(3,2,5)
        ax6 = fig.add_subplot(3,2,6)
        axes = [
            [ax1, ax3, ax5],
            [ax2, ax4, ax6]
        ]
        numerators = [residual, null_residual]
        denominators = [
            1e-16, stellar/1e6, uncertainty
        ]
        powers = [1,1,2]
        labels = [
            r'$10^{-16}\mathrm{W m^{-2} \mu m^{-1}}$',
            'Star contrast (ppm)',
            'Noise contrast'
        ]
        ax1.set_title('Thermal Contrast')
        ax2.set_title('Null Contrast')
        _wl = bin_image(wl.to_value(WL_UNIT).T,BIN_WL, 1,1)[0,:]
        logger.info(f'Binned wavelength has shape {_wl.shape}.')
        _time = bin_image(time.to_value(TIME_UNIT),BIN_TIME, 1,1)[0,:]
        logger.info(f'Binned time has shape {_time.shape}.')
        for _axes, numerator in zip(axes, numerators):
            for ax, denom, label,power in zip(_axes, denominators, labels,powers):
                Z = bin_image(numerator,BIN_WL,BIN_TIME,1) / bin_image(denom,BIN_WL,BIN_TIME,power)
                vminmax = np.max(np.abs(Z))
                im1 = ax.pcolormesh(_wl, _time, Z, rasterized=True, cmap='bwr',vmin=-vminmax,vmax=vminmax)
                ax.set_xlabel('Wavelength ($\\rm \\mu m$)')
                ax.set_ylabel('Time (days)')
                ax.axvline(CUTOFF_WL.to_value(WL_UNIT), ls='--', c='k')
                ax.axvline(CHI2_WL.to_value(WL_UNIT), ls='--', c='k')
                regions = [_wl < CUTOFF_WL.to_value(WL_UNIT), (_wl>=CUTOFF_WL.to_value(WL_UNIT)) & (_wl<CHI2_WL.to_value(WL_UNIT)), _wl>=CHI2_WL.to_value(WL_UNIT)]
                for reg in regions:
                    Zreg = Z[:,reg]
                    mean = np.mean(Zreg)
                    stdev = np.std(Zreg)
                    rms = np.sqrt(np.mean(Zreg**2))
                    ma = np.mean(np.abs(Zreg))
                    s = f'Mean: {mean:.1f}\n' \
                        + f'Std:  {stdev:.1f}\n' \
                        + f'rms:  {rms:.1f}\n' \
                        + f'mae:  {ma:.1f}'
                    ax.text(_wl[reg][0],6,s)
                    
                cbar1 = fig.colorbar(im1, ax=ax, orientation='vertical', shrink=0.8,label=label)
    
    
    with figure_context(figsize=(6, 4)) as fig:
        max_bin_wl = 3
        max_bin_time = 2
        ax = fig.add_subplot(1,1,1)
        im = np.zeros((max_bin_wl,max_bin_time))
        temp_array = np.linspace(0.05, 0.99, 100)
        log_eps_array = (temp_to_log_epsilon(temp_array))
        for i in range(max_bin_wl):
            for j in range(max_bin_time):
                data_residual = bin_image(null_residual,i+1,j+1,1)
                data_uncertainty = bin_image(uncertainty,i+1,j+1,2)
                reg = bin_image(wl.to_value(WL_UNIT),i+1,1,1)[0,:] > CHI2_WL.to_value(WL_UNIT)
                chi2_arr = np.zeros_like(temp_array)
                for k,log_eps in enumerate(log_eps_array):
                    model_thermal = interpolator([log_eps])[0, :, :].T
                    model_reconstruction = vpie.get_reconstruction(
                        model_thermal,
                        coeffs_null,
                        s_null
                    )
                    model_residual = model_reconstruction - model_thermal
                    model_residual = bin_image(model_residual,i+1,j+1,1)
                    difference = data_residual - model_residual
                    chi_spec = (difference/data_uncertainty)[:,reg]
                    chi2_red = np.sum(chi_spec**2)/(chi_spec.size+1)
                    chi2_arr[k] = chi2_red
                ax.plot(temp_array,chi2_arr,label=f'({i}, {j})')
        ax.set_yscale('log')
        ax.legend()        