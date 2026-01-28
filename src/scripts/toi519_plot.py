"""
Make figure for TOI 519 example
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import paths
import VSPEC
import vpie

from toi519_run import get_model
from common import bin_image, figure_context, COLWIDTH

OUTFILE_PREFIX = 'toi519_fig'
FIGSIZE = (COLWIDTH, 0.7*COLWIDTH)

NOISE_SCALE = 1.0
SEED = 10
LABEL_FONT_SIZE = 12

if __name__ == '__main__':
    plt.style.use('bmh')
    rng = np.random.default_rng(SEED)
    model = get_model()
    data = VSPEC.PhaseAnalyzer.from_model(model)

    wl = data.wavelength.to_value(u.um)
    time = data.time.to_value(u.day)
    flux_unit = u.Unit('W m-2 um-1')

    thermal = data.thermal.to_value(flux_unit)
    noise = data.noise.to_value(flux_unit) * NOISE_SCALE
    total = data.total.to_value(flux_unit)
    observed = total + rng.normal(0, noise)
    
    with figure_context(figsize=FIGSIZE) as fig:
        ax = fig.add_subplot(1, 1, 1)
        im = ax.pcolormesh(wl, time, thermal.T, rasterized=True, cmap='afmhot_r')
        ax.set_xlabel('Wavelength ($\\mathrm{ \\mu m}$)')
        ax.set_ylabel('Time (days)')
        ax.set_facecolor('w')
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label('Thermal emission ($\\mathrm{W m^{-2} \\mu m^{-1}}$)')
        fig.tight_layout()
        fig.savefig(paths.figures / f'{OUTFILE_PREFIX}_thermal.pdf')

    cutoff_index = np.argwhere(wl > 1.5)[0][0]
    print(f'Cutoff index: {cutoff_index}')
    s, coeffs, f_rec = vpie.vpie.get_vpie(
        observed.T,
        noise.T,
        cutoff_index=cutoff_index,
        use_mean_error=True,
        ic_string='AIC',
        max_basis_size=2
    )
    
    with figure_context(figsize=FIGSIZE) as fig:
        ax = fig.add_subplot(1, 1, 1)
        tax = ax.twinx()
        ax.plot(
            data.time.to_value(u.day),
            data.lightcurve('star', (0, -1), 'max'),
            lw=2,
            color='xkcd:cerulean',
            label='White light',
        )
        _, q = coeffs.shape
        colors = ['xkcd:lavender', 'xkcd:golden rod', 'xkcd:forest green']
        for i in range(q):
            tax.plot(time, coeffs[:, i], lw=2, label=f'$a_{i+1}$', color=colors[i])
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = tax.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left')
        ax.set_xlabel('Time (days)')
        ax.set_facecolor('w')
        ax.grid(False)
        tax.grid(False)
        tax.set_ylabel('Spectral basis coefficients')
        ax.set_ylabel('Star white light (normalized)')
        fig.tight_layout()
        fig.savefig(paths.figures / f'{OUTFILE_PREFIX}_coeffs.pdf')


    residuals = (f_rec - observed.T)/observed.T * 100
    vminmax = np.max(np.abs(residuals))

    
    with figure_context(figsize=FIGSIZE) as fig:
        ax = fig.add_subplot(1, 1, 1)
        im = ax.pcolormesh(
            wl, time, residuals,
            rasterized=True,
            vmin=-vminmax, vmax=vminmax,
            cmap='bwr'
        )
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label('Residual (%)')
        ax.set_xlabel('Wavelength ($\\rm \\mu m$)')
        ax.set_ylabel('Time (days)')
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(paths.figures / f'{OUTFILE_PREFIX}_residuals_unbinned.pdf')
    
    wl_bin = 6
    time_bin = 4
    binned_residuals = bin_image(residuals, wl_bin, time_bin,1)
    binned_wl = bin_image(wl, wl_bin, 1,1)[0,:]
    binned_time = bin_image(time, time_bin, 1,1)[0,:]
    vminmax = np.max(np.abs(binned_residuals))
    
    with figure_context(figsize=FIGSIZE) as fig:
        ax = fig.add_subplot(1, 1, 1)
        im = ax.pcolormesh(
            binned_wl, binned_time, binned_residuals,
            rasterized=True,
            vmin=-vminmax, vmax=vminmax,
            cmap='bwr'
        )
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label('Residual (%)')
        ax.set_xlabel('Wavelength ($\\rm \\mu m$)')
        ax.set_ylabel('Time (days)')
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(paths.figures / f'{OUTFILE_PREFIX}_residuals_binned.pdf')
    
    