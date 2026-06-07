from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import libpypsg as psg


from vpie import vpie
import VSPEC

import paths
from common import COLWIDTH

from gj876_grid import get_interp
from gj876_run import get_model

PREFIX = 'chi2_test'
IC = 'BIC'
FIGSIZE = (1*COLWIDTH, 1*COLWIDTH)
MAX_BASIS = 2
FLUX_UNIT = u.Unit('W m-2 um-1')
NOISE_SCALE = 1.0
CHI2_NOISE_SCALE = np.sqrt(1)
THERMAL_SCALE = 1.0
SEED = 11
CUTOFF_WL = 1.0*u.um
CHI2_WL = 4.0*u.um
BIN_WL = 6
BIN_TIME = 3
LEGEND_TEXT_SIZE = 8
AXIS_TEXT_SIZE = 10
INDEX_TO_PLOT = -1

if __name__ in '__main__':
    plt.style.use('bmh')
    interpolator = get_interp()
    """
    Takes in [log_epsilon]
    """
    model = get_model()
    model.params.star.period = 4*u.day
    model.params.header.data_path = Path(__file__).parent / '.vspec' / 'gj876-chi2-example'
    if not model.directories['all_model'].exists():
        psg.docker.set_url_and_run()
        model._build_directories()
        model.build_planet()
        model.build_spectra()
    data = VSPEC.PhaseAnalyzer.from_model(model)
    wl = data.wavelength
    time = data.time
    cutoff_index = np.argwhere(wl > CUTOFF_WL)[0][0]
    
    def get_residual_and_noise(chi_noise_scale, epsilon):
        # logger.info(
        #     f'Running radius={radius:.2f} Rp with noise scale of {chi_noise_scale:.2f}')
        if epsilon is None:
            _thermal = 0
        else:
            _thermal = THERMAL_SCALE*interpolator([np.log10(epsilon)])[0, :, :].T
        _data = VSPEC.PhaseAnalyzer.from_model(model)
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
        _residual = _total_observed - _f_rec
        return _residual, _uncertainty, _s, _coeffs,_stellar, _thermal if epsilon is not None else np.zeros_like(_residual)
    
    residual_baseline, uncertainty_baseline, s, coeffs,stellar, thermal = get_residual_and_noise(
            CHI2_NOISE_SCALE, 0.05)
    _, _, _, _, _, thermal_wrong = get_residual_and_noise(CHI2_NOISE_SCALE, 1)
    f_rec_no_planet = vpie.get_reconstruction(stellar, coeffs, s)
    residual_no_planet = stellar - f_rec_no_planet
    # binned_residual_baseline = bin_image(residual_baseline,BIN_WL,BIN_TIME,1)
    # binned_thermal_baseline = bin_image(thermal,BIN_WL,BIN_TIME,1)
    # binned_noise_baseline = bin_image(uncertainty_baseline,BIN_WL,BIN_TIME,2)
    # binned_wl = bin_image(wl.to_value(u.um),BIN_WL,1,1)[0,:]
    # binned_time = bin_image(time.to_value(u.day),BIN_TIME,1,1)[0,:]
    # fig = plt.figure(figsize=FIGSIZE)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(time.to_value(u.day), coeffs)
    # plt.show()
    # plt.close(fig)
    
    fig = plt.figure(figsize=FIGSIZE,layout='constrained')
    ax = fig.add_subplot(1, 1, 1)
    # ax.plot(time.to_value(u.day),f_rec_no_planet[:,INDEX_TO_PLOT]-np.min(f_rec_no_planet[:,INDEX_TO_PLOT]),c='C4',ls='--',label='Star-only variations')
    # ax.plot(time.to_value(u.day),thermal_wrong[:,INDEX_TO_PLOT],c='C2',ls='--',label='Thermal Emission Phase Curve \u2013 Incorrect Model')
    ax.plot(time.to_value(u.day),residual_baseline[:,INDEX_TO_PLOT],c='C0',ls='-',label='Data $\\boldsymbol{\\epsilon} + \\boldsymbol{\\delta} \\approx \\boldsymbol{\\delta}$')
    ax.fill_between(
        time.to_value(u.day),
        residual_baseline[:,INDEX_TO_PLOT] - uncertainty_baseline[:,INDEX_TO_PLOT],
        residual_baseline[:,INDEX_TO_PLOT] + uncertainty_baseline[:,INDEX_TO_PLOT],
        color='C0',
        alpha=0.5,
        label='$\\pm 1\\sigma$'
    )
    ax.fill_between(
        time.to_value(u.day),
        residual_baseline[:,INDEX_TO_PLOT] - 2*uncertainty_baseline[:,INDEX_TO_PLOT],
        residual_baseline[:,INDEX_TO_PLOT] + 2*uncertainty_baseline[:,INDEX_TO_PLOT],
        color='C0',
        alpha=0.2,
        label='$\\pm 2\\sigma$'
    )
    ax.plot(time.to_value(u.day),thermal[:,INDEX_TO_PLOT],c='C1',ls='--',label='$\\boldsymbol{f}(\\theta_1)$, $\\theta_1 = \\{\\mathrm{inefficient,}\\,2.5\\,R_\\oplus\\}$')
    for i, k in enumerate(sorted(list(s))):
        a = coeffs[:,i]
        print(k)
        f_k = thermal[k,INDEX_TO_PLOT]
        akfk = a*f_k
        k_set = r",\,".join([str(k) for k in sorted(list(s))])
        label = f'$ -a_k \\,\\boldsymbol{{f}}_k(\\theta_1) $ for $k \\in \\{{{k_set}\\}}$' if i==0 else None
        ax.plot(time.to_value(u.day),-akfk,c='C1',ls=':',label=label)

    model_residual = thermal - vpie.get_reconstruction(thermal, coeffs, s)
    chi2 = (residual_baseline[:,INDEX_TO_PLOT] - model_residual[:,INDEX_TO_PLOT])**2 / uncertainty_baseline[:,INDEX_TO_PLOT]**2
    red_chi2 = np.sum(chi2)/(len(chi2) - 2)
    label='$\\boldsymbol{\\delta}(\\theta_1) = \\boldsymbol{f}(\\theta_1) - \\sum a_k \\,\\boldsymbol{{f}}_k(\\theta_1) $ \t'+f'$\\chi^2_\\mathrm{{red}} = {red_chi2:.1f}$'
    ax.plot(time.to_value(u.day),model_residual[:,INDEX_TO_PLOT],c='C1',label=label)
    model_residual_wrong = thermal_wrong - vpie.get_reconstruction(thermal_wrong, coeffs, s)
    chi2 = (residual_baseline[:,INDEX_TO_PLOT] - model_residual_wrong[:,INDEX_TO_PLOT])**2 / uncertainty_baseline[:,INDEX_TO_PLOT]**2
    red_chi2 = np.sum(chi2)/(len(chi2) - 2)
    label='$\\boldsymbol{\\delta}(\\theta_2)$, $\\theta_2 = \\{\\mathrm{mod.\\;efficient,}\\,2.5\\,R_\\oplus\\}$\t'+f'$\\chi^2_\\mathrm{{red}} = {red_chi2:.1f}$'
    ax.plot(time.to_value(u.day),model_residual_wrong[:,INDEX_TO_PLOT],c='C2',label=label)
    scale = (3/2.5)**2
    model_residual = thermal*scale - vpie.get_reconstruction(thermal*scale, coeffs, s)
    chi2 = (residual_baseline[:,INDEX_TO_PLOT] - model_residual[:,INDEX_TO_PLOT])**2 / uncertainty_baseline[:,INDEX_TO_PLOT]**2
    red_chi2 = np.sum(chi2)/(len(chi2) - 2)
    label='$\\boldsymbol{\\delta}(\\theta_3)$, $\\theta_3 = \\{\\mathrm{inefficient,}\\,3.0\\,R_\\oplus\\}$\t'+f'$\\chi^2_\\mathrm{{red}} = {red_chi2:.1f}$'
    ax.plot(time.to_value(u.day),model_residual[:,INDEX_TO_PLOT],c='C3',label=label)

    


    # temps = [0.5,0.99]
    # labels = [
    #     'Moderate heat redistribution',
    #     'Full heat redistribution',
    # ]
    
    # for temp, label in zip(temps, labels):
    #     log_epsilon = temp_to_log_epsilon([temp])[0]
    #     residual, uncertainty, s, coeffs, thermal = get_residual_and_noise(
    #         CHI2_NOISE_SCALE, 10**log_epsilon)
    #     binned_residual = bin_image(residual,BIN_WL,BIN_TIME,1)
    #     binned_thermal = bin_image(thermal,BIN_WL,BIN_TIME,1)
    #     binned_noise = bin_image(uncertainty,BIN_WL,BIN_TIME,2)
    #     chi2_spec = (binned_residual-binned_residual_baseline)**2/binned_noise_baseline**2
    #     chi2_lw = chi2_spec[:, binned_wl > CHI2_WL.to_value(u.um)]
    #     red_chi2 = np.sum(chi2_lw) / (chi2_lw.size-2)
    #     logger.info(f'{temp:.2f}: {red_chi2:.2f}')
    #     _label = f'{label} ($\\chi^2_{{\\rm red}}={red_chi2:.2f}$)'
    #     ax.plot(binned_time,binned_thermal[:,INDEX_TO_PLOT],label=_label)
    ax.set_xlabel('Time (days)', fontsize=AXIS_TEXT_SIZE, fontfamily='serif')
    ax.set_ylabel('Flux @ LW ($\\mathrm{W m^{-2} \\mu m^{-1}}$)', fontsize=AXIS_TEXT_SIZE, fontfamily='serif')
    # ax.plot(binned_time,binned_noise_baseline[:,INDEX_TO_PLOT],color='k',ls='--',label='Data uncertainty')
    ax.legend(prop={'size': LEGEND_TEXT_SIZE, 'family': 'serif'},bbox_to_anchor=(0.5, 1.05),loc='lower center')
    ax.set_facecolor('w')
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0],ylims[1]*1.3)
    ax.set_title('')
    # fig.tight_layout()
    fig.savefig(paths.figures / 'chi2_test.pdf')