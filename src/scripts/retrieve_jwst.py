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
from scipy.optimize import minimize_scalar

from vpie import vpie
import VSPEC

import paths
from jwst_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from run_jwst import get_model, get_temperature_ratio as epsilon_to_temp, PLANET as PLANET_PARAMS

TRUE_TEMPERATURE_RATIO = 0.5
TRUE_EPSILON = temp_to_log_epsilon([TRUE_TEMPERATURE_RATIO])
NOISE_SCALE = 1.0
CHI2_NOISE_SCALE = np.sqrt(2.41)
THERMAL_SCALE = 4.0
SEED = 100
FLUX_UNIT = u.Unit('W m-2 um-1')
CUTOFF_WL = 1*u.um
OUTLIER_PERCENTILE = 100

@contextlib.contextmanager
def figure_context(*args, **kwargs):
    fig:plt.Figure = plt.figure(*args,**kwargs)
    yield fig
    plt.close(fig)
    

if __name__ in '__main__':
    plt.style.use('bmh')
    interpolator = get_interp()
    """
    Takes in [log_epsilon]
    """
    thermal = THERMAL_SCALE*interpolator([np.log10(TRUE_EPSILON)])[0,:,:].T #m x n
    data = VSPEC.PhaseAnalyzer.from_model(get_model())
    wl = data.wavelength
    time = data.time
    
    with figure_context(figsize=(6,4)) as fig:
        ax:plt.Axes = fig.subplots(1,1)
        im=ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            thermal,cmap='gist_heat_r',
            rasterized=True
        )
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        ax.set_xscale('log')
        cbar = fig.colorbar(im,ax=ax)
        cbar.set_label('$F_\\lambda/[\\rm W m^{-2} \\mu m^{-1}]$')
        fig.tight_layout()
        fig.savefig(paths.figures / 'jwst_retrieval_thermal.pdf')
    
    rng = np.random.default_rng(SEED)
    stellar = data.star.T.to_value(FLUX_UNIT)
    true_noise = data.noise.T.to_value(FLUX_UNIT)
    noise = true_noise * NOISE_SCALE
    total_true = stellar + thermal
    scatter = rng.normal(loc=0,scale = noise)
    total_observed = total_true + scatter
    
    with figure_context(figsize=(6,4)) as fig:
        ax:plt.Axes = fig.subplots(1,1)
        z = 1e6*(scatter/total_true)
        vminmax = np.max(np.abs(z))
        im=ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            z,cmap='bwr',
            rasterized=True,
            vmin = -vminmax, vmax = vminmax
        )
        ax.set_title(f'Mean {np.mean(z):.1f} ppm, Stdev {np.std(z):.1f} ppm')
        ax.set_xscale('log')
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        cbar = fig.colorbar(im,ax=ax)
        cbar.set_label('Noise (ppm)')
        fig.tight_layout()
        fig.savefig(paths.figures / 'jwst_retrieval_scatter.pdf')
    
    cutoff_index = np.argwhere(wl>CUTOFF_WL)[0][0]
    logger.info(f'For a short-wave cutoff of {CUTOFF_WL}, we choose a cutoff index of {cutoff_index}. Total wl axis size is {wl.size}')
    s, coeffs, f_rec = vpie.get_vpie(
        total_observed,
        noise,
        cutoff_index,
        True
    )
    
    residual = f_rec - total_observed
    
    with figure_context(figsize=(6,4)) as fig:
        ax:plt.Axes = fig.subplots(1,1)
        for i,_s in enumerate(s):
            ax.plot(time,coeffs[:,i],label=f'$c_{{{_s}}}$')
            ax.set_xlabel('$t/[\\rm hr]$')
            fig.savefig(paths.figures / 'jwst_retrieval_coefficients.png')
    
    
    def get_residual_and_noise(distance,fiducial_distance=10,chi_noise_scale=1.0,epsilon=TRUE_EPSILON):
        logger.info(f'Running distance={distance} pc with noise scale of {chi_noise_scale:.2f}')
        distance_noise_scale = (distance/fiducial_distance)**2
        _thermal = THERMAL_SCALE*interpolator([np.log10(epsilon)])[0,:,:].T
        _data = VSPEC.PhaseAnalyzer.from_model(get_model())
        _rng = np.random.default_rng(SEED)
        _stellar = _data.star.T.to_value(FLUX_UNIT)
        _noise = _data.noise.T.to_value(FLUX_UNIT) * NOISE_SCALE * distance_noise_scale
        _total_true = _stellar + _thermal
        _scatter = _rng.normal(loc=0,scale = _noise)
        _noise = _noise * chi_noise_scale
        _total_observed = _total_true + _scatter
        _cutoff_index = np.argwhere(wl>CUTOFF_WL)[0][0]
        _, _, _f_rec = vpie.get_vpie(
            _total_observed,
            _noise,
            _cutoff_index,
            True
        )
        _residual = _f_rec - _total_observed
        return _residual, _noise
        
        
        
        
    
    with figure_context(figsize=(6,4)) as fig:
        ax:plt.Axes = fig.subplots(1,1)
        z = residual / noise
        vminmax = np.max(np.abs(z))
        im=ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            z,cmap='bwr',
            rasterized=True,
            vmin = -vminmax, vmax = vminmax
        )
        ax.set_title(f'Short: {np.mean(z[:,:cutoff_index]):.1f} $\\pm$ {np.std(z[:,:cutoff_index]):.1f} ppm | Long: {np.mean(z[:,cutoff_index:]):.1f} $\\pm$ {np.std(z[:,cutoff_index:]):.1f} ppm')
        ax.axvline(CUTOFF_WL.to_value(u.um),ls='--',c='k')
        ax.set_xscale('log')
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        cbar = fig.colorbar(im,ax=ax)
        cbar.set_label('Noise (ppm)')
        fig.tight_layout()
        fig.savefig(paths.figures / 'jwst_retrieval_residual_scatter.pdf')
        
        
    true_thermal_reconstruction = vpie.get_reconstruction(
        thermal,
        coeffs,
        s
    )
    true_thermal_residual = true_thermal_reconstruction - thermal
    
    with figure_context(figsize=(6,4)) as fig:
        ax:plt.Axes = fig.subplots(1,1)
        vminmax = np.max(np.abs(true_thermal_residual))
        im=ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            true_thermal_residual,cmap='bwr',
            rasterized=True,
            vmin=-vminmax,vmax=vminmax
        )
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        ax.set_xscale('log')
        cbar = fig.colorbar(im,ax=ax)
        cbar.set_label('Thermal Reconstruction $F_\\lambda/[\\rm W m^{-2} \\mu m^{-1}]$')
        fig.tight_layout()
        fig.savefig(paths.figures / 'jwst_retrieval_true_reconstruction.pdf')
        
        
    with figure_context(figsize=(6,4)) as fig:
        ax:plt.Axes = fig.subplots(1,1)
        grid_thermal = THERMAL_SCALE*interpolator([np.log10(TRUE_EPSILON)-0.])[0,:,:].T
        grid_reconstruction = vpie.get_reconstruction(
            grid_thermal,
            coeffs,
            s
        )
        grid_residual = grid_reconstruction - grid_thermal
        difference = grid_residual - residual
        chi_sq = (
            difference**2/noise**2
        )
        # chi_sq = median_filter(chi_sq,5)
        outlier = chi_sq > np.percentile(chi_sq,100)
        chi_sq[outlier] = np.nan
        im=ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            chi_sq,
            rasterized=True,
            # norm=LogNorm()
        )
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        ax.set_xscale('log')
        ax.axvline(CUTOFF_WL.to_value(u.um),ls='--',c='k')
        cbar = fig.colorbar(im,ax=ax)
        cbar.set_label('$\\chi^2$')
        fig.tight_layout()
        fig.savefig(paths.figures / 'jwst_retrieval_chi_sq_true.png')
        
    with figure_context(figsize=(6,4)) as fig:
        ax:plt.Axes = fig.subplots(1,1)
        grid_thermal = THERMAL_SCALE*interpolator([np.log10(TRUE_EPSILON)-0.])[0,:,:].T
        grid_reconstruction = vpie.get_reconstruction(
            grid_thermal,
            coeffs,
            s
        )
        grid_residual = grid_reconstruction - grid_thermal
        difference = grid_residual - residual
        chi = (
            difference/noise
        )
        vminmax = np.max(np.abs(chi))
        im=ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            chi,
            rasterized=True,
            vmin=-vminmax,vmax=vminmax
            # norm=LogNorm()
        )
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        ax.set_xscale('log')
        ax.set_title(f'Mean: {np.mean(chi):.2f} Std: {np.std(chi):.2f} $T_{{\\rm night}} / T_{{\\rm day}}$: {TRUE_TEMPERATURE_RATIO:.2f}')
        ax.axvline(CUTOFF_WL.to_value(u.um),ls='--',c='k')
        cbar = fig.colorbar(im,ax=ax)
        cbar.set_label('$\\chi$')
        fig.tight_layout()
        fig.savefig(paths.figures / 'jwst_retrieval_chi_true.png')
        
        
    
    temp_array = np.linspace(0.05,0.99,100)
    log_eps_array = (temp_to_log_epsilon(temp_array))
    red_chi_sq_array = []
    red_chi_sq_short_array = []
    red_chi_sq_long_array = []
    for log_eps in log_eps_array:
        grid_thermal = THERMAL_SCALE*interpolator([log_eps])[0,:,:].T
        grid_reconstruction = vpie.get_reconstruction(
            grid_thermal,
            coeffs,
            s
        )
        grid_residual = grid_reconstruction - grid_thermal
        difference = grid_residual - residual
        chi_sq_spec = difference**2/(noise*CHI2_NOISE_SCALE)**2
        # chi_sq_spec = median_filter(chi_sq_spec,5)
        chi_sq = np.sum(chi_sq_spec)
        chi_sq_short = np.sum(chi_sq_spec[:,:cutoff_index])
        red_chi_sq_short = chi_sq_short / (chi_sq_spec[:,:cutoff_index]).size
        chi_sq_long = np.sum(chi_sq_spec[:,cutoff_index:])
        red_chi_sq_long = chi_sq_long / (chi_sq_spec[:,cutoff_index:]).size
        red_chi_sq = chi_sq / (chi_sq_spec).size
        red_chi_sq_array.append(red_chi_sq)
        red_chi_sq_short_array.append(red_chi_sq_short)
        red_chi_sq_long_array.append(red_chi_sq_long)
    red_chi_sq_array = np.array(red_chi_sq_array)
    
    with figure_context(figsize=(6,4)) as fig:
        ax:plt.Axes = fig.subplots(1,1)
        im=ax.plot(temp_array,red_chi_sq_array,c='k')
        ax.plot(temp_array,red_chi_sq_short_array,c='r',label='NIR')
        ax.plot(temp_array,red_chi_sq_long_array,c='b',label='MIR')
        ax.set_title(f'Min $\\chi^2_{{\\rm red}}= {np.min(red_chi_sq_array):.2f}$ at $\\log \\epsilon= {log_eps_array[np.argmin(red_chi_sq_array)]:.2f}$')
        ax.set_xlabel('$T_{\\rm night} / T_{\\rm day}$')
        ax.set_ylabel('$\\chi^2_{\\rm red}$')
        ax.set_yscale('log')
        ax.axhline(1,ls='--',c='k')
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.figures / 'jwst_retrieval_red_chi_square.png')
    
    temp_ratios = [0.1,0.5,0.9]
    epsilons = temp_to_log_epsilon(temp_ratios)
    fnames = ['1','5','9']
    for epsilon, fname in zip(epsilons,fnames):
        grid_distance = 4.6
        distance_arr = np.logspace(0.1,0.7,21)
        red_chi_sq_array = np.zeros((distance_arr.size,log_eps_array.size))
        best_radius_array = np.zeros((distance_arr.size,log_eps_array.size))
        for i,dist in enumerate(distance_arr):
            # dist_residual, dist_noise = get_residual_and_noise(dist,fiducial_distance=grid_distance,epsilon=10**epsilon)
            # logger.info(f'Distance: {dist}')
            # for j,log_eps in enumerate(log_eps_array):
            #     grid_thermal = THERMAL_SCALE*interpolator([log_eps])[0,:,:].T
            #     grid_reconstruction = vpie.get_reconstruction(
            #         grid_thermal,
            #         coeffs,
            #         s
            #     )
            #     grid_residual = grid_reconstruction - grid_thermal
            #     difference = grid_residual - dist_residual
            #     chi_sq_spec = difference**2/(dist_noise)**2
            #     chi_sq = np.sum(chi_sq_spec)
            #     red_chi_sq = chi_sq / chi_sq_spec.size
            #     red_chi_sq_array[i,j] = red_chi_sq
            # chi_noise_scale = np.min(red_chi_sq_array[i,:])
            dist_residual, dist_noise = get_residual_and_noise(dist,fiducial_distance=grid_distance,chi_noise_scale=4.47,epsilon=10**epsilon)
            for j,log_eps in enumerate(log_eps_array):
                grid_thermal = THERMAL_SCALE*interpolator([log_eps])[0,:,:].T
                grid_reconstruction = vpie.get_reconstruction(
                    grid_thermal,
                    coeffs,
                    s
                )
                def get_chi_sq(rp:float):
                    grid_residual = grid_reconstruction - grid_thermal
                    difference = rp**2*grid_residual - dist_residual
                    chi_sq_spec = difference**2/(dist_noise)**2
                    chi_sq = np.sum(chi_sq_spec)
                    red_chi_sq = chi_sq / chi_sq_spec.size
                    return red_chi_sq
                
                res = minimize_scalar(get_chi_sq,bounds=(0.,50),method='bounded')
                soln = res.x
                # logger.info(f'Best radius: {soln}')
                best_radius_array[i,j] = soln
                red_chi_sq_array[i,j] = get_chi_sq(soln)
                
                
                # grid_residual = grid_reconstruction - grid_thermal
                # difference = grid_residual - dist_residual
                # chi_sq_spec = difference**2/(dist_noise)**2
                # chi_sq = np.sum(chi_sq_spec)
                # red_chi_sq = chi_sq / chi_sq_spec.size
                # red_chi_sq_array[i,j] = red_chi_sq
            
        best_radius_array = best_radius_array * PLANET_PARAMS.radius.to_value(u.R_earth)
        with figure_context(figsize=(6,4.5)) as fig:
            axes:tuple[plt.Axes,plt.Axes] = fig.subplots(2,1,sharex=True,gridspec_kw={'height_ratios':[1,3]})
            fig.subplots_adjust(hspace=0)
            ax0,ax = axes
            ax0.plot(temp_array,best_radius_array.mean(axis=0),c='k')
            ax0.set_ylabel('$R_\\mathrm{p}/R_\\oplus$')
            ax0.set_ylim(0,10)
            ax0.set_yticks([0,2,4,6,8])
            ax0.set_facecolor('w')
            ax0.axhline(PLANET_PARAMS.radius.to_value(u.R_earth),c='r',ls='--',zorder=-100)
            im=ax.pcolormesh(
                temp_array,distance_arr,(red_chi_sq_array),
                rasterized=True,
                norm=LogNorm()
            )
            ax.set_ylabel('$d/[\\rm pc]$')
            ax.set_xlabel('$T_{\\rm night} / T_{\\rm day}$')
            ax.set_yscale('log')
            ax.grid(False)
            yticks = [1.5,2,3,4,6,8,10,12]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
            ax.axvline(x=float(fname)/10,c='r',ls='--')
            fig.colorbar(im,label='$\\chi^2_{\\rm red}$')
            levels = [1,4,9,16,25]
            fmt = lambda x: f'$\\chi^2_{{\\rm red}} = {x:.0f}$'
            im=ax.contour(
                temp_array,distance_arr,red_chi_sq_array,
                levels=levels,
                colors='k',
                linestyles='dashed'
            )
            ax.clabel(im,im.levels,inline=True,fontsize=10,fmt=fmt)
            # levels=[0.1,0.5,1,2]
            # im=ax.contour(
            #     temp_array,distance_arr,best_radius_array,
            #     levels=levels,
            #     colors='r',
            #     linestyles='-'
            # )
            # fmt = lambda x: f'$R_\mathrm{{p}} = {x:.1f}~R_\odot$'
            # ax.clabel(im,im.levels,inline=True,fontsize=10,fmt=fmt)
            pos = ax.get_position()
            pos0 = ax0.get_position()
            ax0.set_position([pos.x0,pos0.y0,pos.width,pos0.height])
            # fig.tight_layout()
            fig.savefig(paths.figures / f'jwst_retrieval_red_chi_square_distance_{fname}.pdf')
        
    
    