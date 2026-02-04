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
from common import bin_image
from scripts.proxb_grid import get_interp, dt_to_eps as temp_to_log_epsilon
from proxb_run import get_model, PLANET as PLANET_PARAMS

PREFIX = 'proxb'
IC = 'AIC'
MAX_BASIS = 4
TRUE_TEMPERATURE_RATIO = 1.0
TRUE_LOG_EPSILON = temp_to_log_epsilon([TRUE_TEMPERATURE_RATIO]) if TRUE_TEMPERATURE_RATIO != 1.0 else None
NOISE_SCALE = 1.0#/np.sqrt(16)
CHI2_NOISE_SCALE = 1
THERMAL_SCALE = 1.0
SEED = 11
FLUX_UNIT = u.Unit('W m-2 um-1')
CUTOFF_WL = 5*u.um
CHI2_WL = 15*u.um
BIN_WL = 6
BIN_TIME = 4


@contextlib.contextmanager
def figure_context(*args, **kwargs):
    fig: plt.Figure = plt.figure(*args, **kwargs)
    yield fig
    plt.close(fig)


if __name__ in '__main__':
    plt.style.use('bmh')
    interpolator = get_interp()
    """
    Takes in [log_epsilon]
    """
    thermal = THERMAL_SCALE * \
        interpolator([TRUE_LOG_EPSILON])[0, :, :].T if TRUE_LOG_EPSILON is not None else 0* interpolator([0.0])[0, :, :].T  # m x n
    data = VSPEC.PhaseAnalyzer.from_model(get_model())
    wl = data.wavelength
    time = data.time

    with figure_context(figsize=(6, 4)) as fig:
        ax: plt.Axes = fig.subplots(1, 1)
        im = ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            thermal/data.star.T.value * 1e6, cmap='gist_heat_r',
            rasterized=True
        )
        ax.axvline(x=CUTOFF_WL.to_value(u.um), c='r', ls='--')
        ax.axvline(x=CHI2_WL.to_value(u.um), c='r', ls='--')
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        ax.set_xscale('log')
        cbar = fig.colorbar(im, ax=ax)
        # cbar.set_label('$F_\\lambda/[\\rm W m^{-2} \\mu m^{-1}]$')
        cbar.set_label('Thermal Flux (ppm)')
        fig.tight_layout()
        fig.savefig(paths.figures / f'{PREFIX}_retrieval_thermal.pdf')

    rng = np.random.default_rng(SEED)
    stellar = data.star.T.to_value(FLUX_UNIT)
    true_noise = data.noise.T.to_value(FLUX_UNIT)
    noise = true_noise * NOISE_SCALE
    total_true = stellar + thermal
    scatter = rng.normal(loc=0, scale=noise)
    total_observed = total_true + scatter

    with figure_context(figsize=(6, 4)) as fig:
        ax: plt.Axes = fig.subplots(1, 1)
        z = 1e6*(scatter/total_true)
        vminmax = np.max(np.abs(z))
        im = ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            z, cmap='bwr',
            rasterized=True,
            vmin=-vminmax, vmax=vminmax
        )
        ax.set_title(f'Mean {np.mean(z):.1f} ppm, Stdev {np.std(z):.1f} ppm')
        ax.set_xscale('log')
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Noise (ppm)')
        fig.tight_layout()
        fig.savefig(paths.figures / f'{PREFIX}_retrieval_scatter.pdf')

    cutoff_index = np.argwhere(wl > CUTOFF_WL)[0][0]
    long_cutoff_index = np.argwhere(wl > CHI2_WL)[0][0]
    logger.info(
        f'For a short-wave cutoff of {CUTOFF_WL}, we choose a cutoff index of {cutoff_index}. Total wl axis size is {wl.size}')
    logger.info(
        f'For a long-wave cutoff of {CHI2_WL}, we choose a cutoff index of {long_cutoff_index}. Total wl axis size is {wl.size}')
    s, coeffs, f_rec = vpie.get_vpie(
        total_observed,
        noise,
        cutoff_index,
        True,
        IC
    )

    residual = f_rec - total_observed

    with figure_context(figsize=(6, 4)) as fig:
        ax: plt.Axes = fig.subplots(1, 1)
        for i, _s in enumerate(s):
            ax.plot(time, coeffs[:, i], label=f'$c_{{{_s}}}$')
            ax.set_xlabel('$t/[\\rm hr]$')
            fig.savefig(paths.figures / f'{PREFIX}_retrieval_coefficients.png')

    def get_residual_and_noise(aperture, fiducial_aperture, chi_noise_scale, epsilon):
        logger.info(
            f'Running aperture={aperture:.1f} m with noise scale of {chi_noise_scale:.2f}')
        aperture_noise_scale = (aperture/fiducial_aperture)**-2
        nullify = epsilon is None
        _thermal = THERMAL_SCALE*interpolator([np.log10(epsilon)])[0, :, :].T if not nullify else 0
        _data = VSPEC.PhaseAnalyzer.from_model(get_model())
        _rng = np.random.default_rng(SEED)
        _stellar = _data.star.T.to_value(FLUX_UNIT)
        _scatter_mag = _data.noise.T.to_value(
            FLUX_UNIT) * NOISE_SCALE * aperture_noise_scale
        _total_true = _stellar + _thermal
        _scatter = _rng.normal(loc=0, scale=_scatter_mag)
        # logger.info(f'Thermal/scatter: {_thermal[:,cutoff_index]/_noise[:,cutoff_index]}')
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

    with figure_context(figsize=(6, 4)) as fig:
        ax: plt.Axes = fig.subplots(1, 1)
        z = residual / noise
        vminmax = np.max(np.abs(z))
        im = ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            z, cmap='bwr',
            rasterized=True,
            vmin=-vminmax, vmax=vminmax
        )
        ax.set_title(
            f'Short: {np.mean(z[:,:cutoff_index]):.1f} $\\pm$ {np.std(z[:,:cutoff_index]):.1f} ppm | Long: {np.mean(z[:,cutoff_index:]):.1f} $\\pm$ {np.std(z[:,cutoff_index:]):.1f} ppm')
        ax.axvline(CUTOFF_WL.to_value(u.um), ls='--', c='k')
        ax.axvline(CHI2_WL.to_value(u.um), ls='--', c='k')
        ax.set_xscale('log')
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Noise (ppm)')
        fig.tight_layout()
        fig.savefig(paths.figures / f'{PREFIX}_retrieval_residual_scatter.pdf')

    true_thermal_reconstruction = vpie.get_reconstruction(
        thermal,
        coeffs,
        s
    )
    true_thermal_residual = true_thermal_reconstruction - thermal

    with figure_context(figsize=(6, 4)) as fig:
        ax: plt.Axes = fig.subplots(1, 1)
        vminmax = np.max(np.abs(true_thermal_residual))
        im = ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            true_thermal_residual, cmap='bwr',
            rasterized=True,
            vmin=-vminmax, vmax=vminmax
        )
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        ax.set_xscale('log')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            'Thermal Reconstruction $F_\\lambda/[\\rm W m^{-2} \\mu m^{-1}]$')
        fig.tight_layout()
        fig.savefig(paths.figures /
                    f'{PREFIX}_retrieval_true_reconstruction.pdf')

    with figure_context(figsize=(6, 4)) as fig:
        ax: plt.Axes = fig.subplots(1, 1)
        grid_thermal = THERMAL_SCALE * \
            interpolator([TRUE_LOG_EPSILON])[0, :, :].T if TRUE_LOG_EPSILON is not None else 0*interpolator([0])[0, :, :].T
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
        outlier = chi_sq > np.percentile(chi_sq, 100)
        chi_sq[outlier] = np.nan
        im = ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            chi_sq,
            rasterized=True,
            norm=LogNorm()
        )
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        ax.set_xscale('log')
        ax.axvline(CUTOFF_WL.to_value(u.um), ls='--', c='k')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('$\\chi^2$')
        fig.tight_layout()
        fig.savefig(paths.figures / f'{PREFIX}_retrieval_chi_sq_true.png')

    with figure_context(figsize=(6, 4)) as fig:
        ax: plt.Axes = fig.subplots(1, 1)
        grid_thermal = THERMAL_SCALE * \
            interpolator([TRUE_LOG_EPSILON])[0, :, :].T if TRUE_LOG_EPSILON is not None else 0 * interpolator([0])[0, :, :].T
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
        im = ax.pcolormesh(
            wl.to_value(u.um),
            time.to_value(u.hr),
            chi,
            rasterized=True,
            vmin=-vminmax, vmax=vminmax
            # norm=LogNorm()
        )
        ax.set_xlabel('$\\lambda/[\\rm \\mu m]$')
        ax.set_ylabel('$t/[\\rm hr]$')
        ax.set_xscale('log')
        ax.set_title(
            f'Mean: {np.mean(chi):.2f} Std: {np.std(chi):.2f} $T_{{\\rm night}} / T_{{\\rm day}}$: {TRUE_TEMPERATURE_RATIO:.2f}')
        ax.axvline(CUTOFF_WL.to_value(u.um), ls='--', c='k')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('$\\chi$')
        fig.tight_layout()
        fig.savefig(paths.figures / f'{PREFIX}_retrieval_chi_true.png')

    temp_array = np.linspace(0.05, 0.99, 5)
    log_eps_array = [temp_to_log_epsilon([a])[0] if a != 1.0 else None for a in temp_array]
    red_chi_sq_array = []
    red_chi_sq_short_array = []
    red_chi_sq_long_array = []
    red_chi_sq_mid_array = []
    aperture = 2.
    grid_aperture = 2
    aperture_residual, aperture_noise, _s, _coeffs = get_residual_and_noise(
        aperture, fiducial_aperture=grid_aperture, epsilon=10**TRUE_LOG_EPSILON if TRUE_LOG_EPSILON is not None else None, chi_noise_scale=CHI2_NOISE_SCALE)
    aperture_residual = bin_image(aperture_residual, BIN_WL, BIN_TIME, 1)
    aperture_noise = bin_image(aperture_noise, BIN_WL, BIN_TIME, 1)
    _wl = bin_image(wl.to_value(u.um), BIN_WL, 1, 1)[0, :]
    is_long = _wl > CHI2_WL.to_value(u.um)
    is_short = _wl < CUTOFF_WL.to_value(u.um)
    for log_eps in log_eps_array:
        grid_thermal = THERMAL_SCALE*interpolator([log_eps])[0, :, :].T if log_eps is not None else 0*interpolator([0])[0, :, :].T
        grid_reconstruction = vpie.get_reconstruction(
            grid_thermal,
            _coeffs,
            _s
        )
        grid_residual = grid_reconstruction - grid_thermal
        grid_residual = bin_image(grid_residual, BIN_WL, BIN_TIME, 1)
        difference = grid_residual - aperture_residual
        chi_sq_spec = difference**2/(aperture_noise)**2
        chi_sq = np.sum(chi_sq_spec)
        chi_sq_short = np.sum(chi_sq_spec[:, is_short])
        red_chi_sq_short = chi_sq_short / (chi_sq_spec[:, is_short]).size
        chi_sq_long = np.sum(chi_sq_spec[:, is_long])
        red_chi_sq_long = chi_sq_long / (chi_sq_spec[:, is_long]).size
        red_chi_sq = chi_sq / (chi_sq_spec).size
        chi_sq_mid = np.sum(chi_sq_spec[:, (~is_long & ~is_short)])
        red_chi_sq_mid = chi_sq_mid / (chi_sq_spec[:, (~is_long & ~is_short)]).size
        red_chi_sq_array.append(red_chi_sq)
        red_chi_sq_short_array.append(red_chi_sq_short)
        red_chi_sq_long_array.append(red_chi_sq_long)
        red_chi_sq_mid_array.append(red_chi_sq_mid)
    red_chi_sq_array = np.array(red_chi_sq_array)

    with figure_context(figsize=(6, 4)) as fig:
        ax: plt.Axes = fig.subplots(1, 1)
        im = ax.plot(temp_array, red_chi_sq_array, c='k')
        ax.plot(temp_array, red_chi_sq_short_array, c='r', label='NIR')
        ax.plot(temp_array, red_chi_sq_long_array, c='b', label='MIR')
        ax.plot(temp_array, red_chi_sq_mid_array, c='g', label='between')
        ax.set_title(
            f'Min $\\chi^2_{{\\rm red}}= {np.min(red_chi_sq_array):.2f}$ at $\\log \\epsilon= {log_eps_array[np.argmin(red_chi_sq_array)]:.2f}$')
        ax.set_xlabel('$T_{\\rm night} / T_{\\rm day}$')
        ax.set_ylabel('$\\chi^2_{\\rm red}$')
        ax.set_yscale('log')
        ax.axhline(1, ls='--', c='k')
        ax.legend()
        fig.tight_layout()
        fig.savefig(paths.figures / f'{PREFIX}_retrieval_red_chi_square.png')

    temp_ratios = [0.1, 0.5, 0.99]
    temp_ratios = [1.0]
    epsilons = [temp_to_log_epsilon([temp_ratio])[0] if temp_ratio != 1.0 else None for temp_ratio in temp_ratios]
    fnames = ['1', '5', '9']
    fnames = ['9']
    bounds = (0., 50.0)
    

    def is_one(x):
        tol = 1e-3
        return (x < 1+tol) and (x > 1-tol)
    for epsilon, fname, temp in zip(epsilons, fnames, temp_ratios):
        grid_aperture = 2
        aperture_arr = np.logspace(-0.5, 1.05, 11)
        red_chi_sq_array = np.zeros((aperture_arr.size, len(log_eps_array)))
        best_radius_array = np.zeros((aperture_arr.size, len(log_eps_array)))
        for i, aperture in enumerate(aperture_arr):
            lowest_chi_sq = 2.0
            while not is_one(lowest_chi_sq):
                aperture_residual, aperture_noise, _s, _coeffs = get_residual_and_noise(
                    aperture,
                    fiducial_aperture=grid_aperture,
                    epsilon=10**epsilon if epsilon is not None else None,
                    chi_noise_scale=np.sqrt(lowest_chi_sq)
                )
                binned_aperture_residual = bin_image(aperture_residual, BIN_WL, BIN_TIME, 1)
                binned_aperture_noise = bin_image(aperture_noise,BIN_WL, BIN_TIME,2)
                del aperture_residual
                del aperture_noise
                binned_wl = bin_image(wl.to_value(u.um),BIN_WL,1,1)[0,:]
                long_wl = binned_wl >= CHI2_WL.to_value(u.um)
                for j, log_eps in enumerate(log_eps_array):
                    grid_thermal = THERMAL_SCALE * \
                        interpolator([log_eps])[0, :, :].T if log_eps is not None else 0*interpolator([0])[0, :, :].T
                    grid_reconstruction = vpie.get_reconstruction(
                        grid_thermal,
                        _coeffs,
                        _s
                    )
                    grid_residual = grid_reconstruction - grid_thermal
                    binned_grid_residual = bin_image(grid_residual, BIN_WL, BIN_TIME, 1)
                    del grid_residual
                    

                    def get_chi_sq(rp: float):
                        difference = rp**2*binned_grid_residual - binned_aperture_residual
                        chi_sq_spec = difference**2/(binned_aperture_noise)**2
                        chi_sq_spec = chi_sq_spec[:, long_wl]
                        chi_sq = np.sum(chi_sq_spec)
                        red_chi_sq = chi_sq / chi_sq_spec.size
                        return red_chi_sq
                    res = minimize_scalar(
                        get_chi_sq, bounds=bounds, method='bounded')
                    soln = res.x
                    best_radius_array[i, j] = soln
                    red_chi_sq_array[i, j] = get_chi_sq(soln)
                new_lowest_chi_sq = np.min(red_chi_sq_array[i, :])
                logger.info(f'New lowest chi sq: {new_lowest_chi_sq:.2f}')
                if is_one(new_lowest_chi_sq):
                    lowest_chi_sq = new_lowest_chi_sq
                else:
                    lowest_chi_sq = lowest_chi_sq * new_lowest_chi_sq

        best_radius_array = best_radius_array * \
            PLANET_PARAMS.radius.to_value(u.R_earth)
        with figure_context(figsize=(6, 4.5)) as fig:
            axes: tuple[plt.Axes, plt.Axes] = fig.subplots(
                2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
            fig.subplots_adjust(hspace=0)
            ax0, ax = axes
            ax0.plot(temp_array, best_radius_array.mean(axis=0), c='k')
            ax0.set_ylabel('$R_\\mathrm{p}/R_\\oplus$')
            ax0.set_ylim(0, 3.1)
            ax0.set_yticks([0, 1, 2, 3])
            ax0.set_facecolor('w')
            ax0.axhline(PLANET_PARAMS.radius.to_value(
                u.R_earth), c='r', ls='--', zorder=-100)
            im = ax.pcolormesh(
                temp_array, aperture_arr, (red_chi_sq_array),
                rasterized=True,
                norm=LogNorm()
            )
            ax.set_ylabel('Aperture Size [$\\mathrm{m}$]')
            ax.set_xlabel('$T_{\\rm night} / T_{\\rm day}$')
            ax.set_yscale('log')
            ax.grid(False)
            yticks = [0.5, 1, 2, 4, 8]
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticks)
            ax.axvline(x=temp, c='r', ls='--')
            fig.colorbar(im, label='$\\chi^2_{\\rm red}$')
            levels = [1, 4, 9, 16, 25]
            def fmt(x): return f'$\\chi^2_{{\\rm red}} = {x:.0f}$'
            im = ax.contour(
                temp_array, aperture_arr, red_chi_sq_array,
                levels=levels,
                colors='k',
                linestyles='dashed'
            )
            ax.clabel(im, im.levels, inline=True, fontsize=10, fmt=fmt)
            # levels=[0.1,0.5,1,2]
            # im=ax.contour(
            #     temp_array,distance_arr,best_radius_array,
            #     levels=levels,
            #     colors='w',
            #     linestyles='-'
            # )
            # fmt = lambda x: f'$R_\mathrm{{p}} = {x:.1f}~R_\\mathrm{{J}}$'
            # ax.clabel(im,im.levels,inline=True,fontsize=10,fmt=fmt)
            pos = ax.get_position()
            pos0 = ax0.get_position()
            ax0.set_position([pos.x0, pos0.y0, pos.width, pos0.height])
            # fig.tight_layout()
            fig.savefig(
                paths.figures / f'{PREFIX}_retrieval_red_chi_square_distance_{fname}.pdf')
