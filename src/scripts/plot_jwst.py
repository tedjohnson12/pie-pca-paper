"""
Make figure for JWST example
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import paths
import VSPEC
import vpie

from run_jwst import get_model

OUTFILE = paths.figures / 'jwst.pdf'
NOISE_SCALE = 0.10
SEED = 10
QUARTER_PERIOD = 9

if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    model = get_model()
    data = VSPEC.PhaseAnalyzer.from_model(model)
    
    fig = plt.figure(figsize = (4.25,9))
    nrow = 4
    ax1 = fig.add_subplot(nrow, 1, 1)
    
    wl = data.wavelength.to_value(u.um)
    time = data.time.to_value(u.day)
    flux_unit = u.Unit('W m-2 um-1')
    
    thermal = data.thermal.to_value(flux_unit)
    noise = data.noise.to_value(flux_unit) * NOISE_SCALE
    total = data.total.to_value(flux_unit)
    observed = total + rng.normal(0,noise)
    
    im1 = ax1.pcolormesh(wl,time,thermal.T,rasterized=True,cmap='magma')
    ax1.set_xlabel('Wavelength ($\\rm \\mu m$)')
    ax1.set_ylabel('Time (days)')
    ax1.text(-0.17,1.1,transform=ax1.transAxes,ha='right',va='top',s='a)',fontsize=12,fontweight='bold')
    
    cbar1 = fig.colorbar(im1,ax=ax1,orientation='vertical',shrink=0.8)
    cbar1.set_label('Thermal flux ($\\rm W m^{-2} \\mu m^{-1}$)')
    
    cutoff_index = np.argwhere(wl > 0.8)[0][0]
    print(f'Cutoff index: {cutoff_index}')
    s, coeffs, f_rec = vpie.vpie.get_vpie(
        observed.T,
        noise.T,
        cutoff_index=cutoff_index,
        use_mean_error=True,
    )
    
    ax2 = fig.add_subplot(nrow, 1, 2)
    i_day = QUARTER_PERIOD*3
    i_night = QUARTER_PERIOD
    ax2.plot(wl,data.spectrum('thermal',i_day,False).to_value(flux_unit),lw=2,label='Day',c='xkcd:goldenrod')
    ax2.plot(wl,data.spectrum('thermal',i_night,False).to_value(flux_unit),lw=2,label='Night',c='xkcd:periwinkle')
    ax2.set_xlabel('Wavelength ($\\rm \\mu m$)')
    ax2.set_ylabel('Thermal flux ($\\rm W m^{-2} \\mu m^{-1}$)')
    ax2.legend()
    ax2.text(-0.17,1.1,transform=ax2.transAxes,ha='right',va='top',s='b)',fontsize=12,fontweight='bold')
    
    ax3 = fig.add_subplot(nrow, 1, 3)
    ax3b = ax3.twinx()
    ax3.plot(time,data.lightcurve('star',(0,-1),'max'),lw=2,color='xkcd:cerulean')
    ax3b.plot(time,coeffs[:,0], lw=2,label='$a_1$',color='xkcd:pale orange')
    ax3b.plot(time,coeffs[:,1], lw=2,label='$a_2$',color='xkcd:grass')
    ax3b.legend()
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Star white light ($\\rm W m^{-2}$)')
    ax3b.set_ylabel('Spectral basis coefficients')
    ax3.text(-0.17,1.1,transform=ax3.transAxes,ha='right',va='top',s='c)',fontsize=12,fontweight='bold')
    
    ax4 = fig.add_subplot(nrow, 1, 4)
    
    residuals = (f_rec - observed.T)/observed.T * 1e6
    vminmax = np.max(np.abs(residuals))
    
    im4 = ax4.pcolormesh(wl,time,residuals,rasterized=True,vmin=-vminmax,vmax=vminmax,cmap='bwr')
    cbar4 = fig.colorbar(im4,ax=ax4,orientation='vertical',shrink=0.8)
    cbar4.set_label('Residual ($\\rm W m^{-2} \\mu m^{-1}$)')
    
    ax4.set_xlabel('Wavelength ($\\rm \\mu m$)')
    ax4.set_ylabel('Time (days)')
    ax4.text(-0.17,1.1,transform=ax4.transAxes,ha='right',va='top',s='d)',fontsize=12,fontweight='bold')
    
    fig.tight_layout()
    fig.subplots_adjust(left=0.2)
    fig.savefig(OUTFILE)

    