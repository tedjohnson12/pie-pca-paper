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

OUTFILE = paths.figures / 'toi519.pdf'
NOISE_SCALE = 1.0
SEED = 10
QUARTER_PERIOD = 15
LABEL_FONT_SIZE = 12

if __name__ == '__main__':
    plt.style.use('bmh')
    rng = np.random.default_rng(SEED)
    model = get_model()
    data = VSPEC.PhaseAnalyzer.from_model(model)

    fig = plt.figure(figsize=(4.25, 9))
    nrow = 4
    ax1 = fig.add_subplot(nrow, 1, 1)

    wl = data.wavelength.to_value(u.um)
    time = data.time.to_value(u.day)
    flux_unit = u.Unit('W m-2 um-1')

    thermal = data.thermal.to_value(flux_unit)
    noise = data.noise.to_value(flux_unit) * NOISE_SCALE
    total = data.total.to_value(flux_unit)
    observed = total + rng.normal(0, noise)

    # def bin_by_two(arr):
    #     if arr.ndim == 1:
    #         return (arr[::2]+arr[1::2])/2
    #     a = arr[:,::2]
    #     b = arr[:,1::2]
    #     return (a+b)/2
    # thermal = bin_by_two(thermal)
    # noise = bin_by_two(noise)/np.sqrt(2)
    # total = bin_by_two(total)
    # observed = bin_by_two(observed)
    # time = bin_by_two(time)

    im1 = ax1.pcolormesh(wl, time, thermal.T, rasterized=True, cmap='afmhot_r')
    ax1.set_xlabel('Wavelength ($\\mathrm{ \\mu m}$)')
    ax1.set_ylabel('Time (days)')
    ax1.set_facecolor('w')
    ax1.grid(False)
    ax1.text(-0.23, 1.1, transform=ax1.transAxes, ha='right',
             va='top', s='a)', fontsize=12, fontweight='bold')

    cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.8)
    cbar1.set_label('Thermal flux ($\\mathrm{W m^{-2} \\mu m^{-1}}$)')

    cutoff_index = np.argwhere(wl > 1.5)[0][0]
    print(f'Cutoff index: {cutoff_index}')
    s, coeffs, f_rec = vpie.vpie.get_vpie(
        observed.T,
        noise.T,
        cutoff_index=cutoff_index,
        use_mean_error=True,
        ic_string='AIC',
        max_basis_size=None
    )

    ax2 = fig.add_subplot(nrow, 1, 2)
    i_day = QUARTER_PERIOD*3
    i_night = QUARTER_PERIOD
    ax2.plot(wl, data.spectrum('thermal', i_day, False).to_value(
        flux_unit), lw=2, label='Day', c='xkcd:goldenrod')
    ax2.plot(wl, data.spectrum('thermal', i_night, False).to_value(
        flux_unit), lw=2, label='Night', c='xkcd:periwinkle')
    ax2.set_xlabel('Wavelength ($\\rm \\mu m$)')
    ax2.set_ylabel('Thermal flux ($\\rm W m^{-2} \\mu m^{-1}$)')
    ax2.legend()
    ax2.set_facecolor('w')
    ax2.grid(False)
    ax2.text(-0.23, 1.1, transform=ax2.transAxes, ha='right',
             va='top', s='b)', fontsize=12, fontweight='bold')

    ax3 = fig.add_subplot(nrow, 1, 3)
    ax3b = ax3.twinx()
    ax3.plot(data.time.to_value(u.day), data.lightcurve(
        'star', (0, -1), 'max'), lw=2, color='xkcd:cerulean')
    ax3b.plot(time, coeffs[:, 0], lw=2,
              label='$a_1$', color='xkcd:pale orange')
    ax3b.plot(time, coeffs[:, 1], lw=2, label='$a_2$', color='xkcd:grass')
    ax3b.plot(time,coeffs[:,2], lw=2,label='$a_3$',color='xkcd:lavender')
    ax3b.legend()
    ax3.set_xlabel('Time (days)')
    ax3.set_facecolor('w')
    ax3.grid(False)
    ax3b.grid(False)
    ax3.set_ylabel('Star white light ($\\rm W m^{-2}$)')
    ax3b.set_ylabel('Spectral basis coefficients')
    ax3.text(-0.23, 1.15, transform=ax3.transAxes, ha='right',
             va='top', s='c)', fontsize=12, fontweight='bold')

    ax4 = fig.add_subplot(nrow, 1, 4)

    residuals = (f_rec - observed.T)/observed.T * 100
    vminmax = np.max(np.abs(residuals))

    im4 = ax4.pcolormesh(wl, time, residuals, rasterized=True,
                         vmin=-vminmax, vmax=vminmax, cmap='bwr')
    cbar4 = fig.colorbar(im4, ax=ax4, orientation='vertical', shrink=0.8)
    cbar4.set_label('Residual (%)')

    ax4.set_xlabel('Wavelength ($\\rm \\mu m$)')
    ax4.set_ylabel('Time (days)')
    ax4.set_facecolor('w')
    ax4.grid(False)
    ax4.text(-0.23, 1.1, transform=ax4.transAxes, ha='right',
             va='top', s='d)', fontsize=12, fontweight='bold')
    fig.text(0.5,0.98,'TOI-519 b',ha='center',va='center',fontsize=16,fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(left=0.2)
    fig.savefig(OUTFILE)

    # plt.close(fig)
    
    # Z = (f_rec - observed.T)/noise.T
    # plt.imshow(Z[:,:cutoff_index],aspect='auto')
    # plt.colorbar()
    # plt.title(np.std(Z[:,:cutoff_index]))
    
    # plt.savefig(paths.figures / 'test.png')