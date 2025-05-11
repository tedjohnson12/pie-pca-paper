from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from time import sleep
from matplotlib.cm import ScalarMappable
from matplotlib.colors import CenteredNorm

import libpypsg as psg
from GridPolator import GridSpectra
import vpie

from colors import cm_teal_cream_orange
import paths

cmap = cm_teal_cream_orange
BASE_CFG = paths.static / 'base.cfg'
OUTPATH = paths.figures / 'fig_3d3.pdf'

PL_MULTIPLIER = 1e4
FONTSIZE = 14
LG_FONTSIZE = 12
WL_SHORT = 1.5*u.um

TEFF_PHOT = np.array([3000])
TEFF_SPOT = np.array([2600])

TEFFS = [2600,2700,2800,2900,3000]

phase = np.linspace(0,2*np.pi,20)
spot_frac = 0.7 + 0.2 * np.cos(phase/2)
tsurf = (600 + 200 * np.sin(phase))*u.K


if __name__ in "__main__":
    psg.docker.set_url_and_run()
    
    cfg = psg.PyConfig.from_file(BASE_CFG)
    spec = GridSpectra.from_vspec(
        w1=cfg.telescope.range1.value,
        w2=cfg.telescope.range2.value,
        resolving_power=cfg.telescope.resolution.value.value,
        teffs=TEFFS,
        fail_on_missing=True
    )
    
    fig = plt.figure(figsize=(3, 3.5))
    ax1 = fig.add_subplot(1, 1, 1,projection='3d')
    fdat = []
    
    for phi,fp,ts in zip(phase,spot_frac,tsurf):
        _cfg = cfg
        _cfg.surface.temperature = ts
        rad = psg.APICall(cfg,'rad')().rad
        sleep(0.2)
        color = cmap(phi/2/np.pi)
        y = rad.wl.to_value(u.um)
        f_planet = rad['Thermal'].to_value(u.W/u.m**2/u.um) * PL_MULTIPLIER
        f_star = spec.evaluate([TEFF_SPOT])*fp + spec.evaluate([TEFF_PHOT])*(1-fp)
        f_star = f_star[0]
        f_star = f_star * (cfg.target.star_radius.value**2/cfg.geometry.observer_altitude.value**2).to_value(u.dimensionless_unscaled)
        f_total = f_planet + f_star
        fdat.append(f_total)
        
        z = np.log10(f_total)
        x = np.ones_like(y) * phi
        ax1.plot(x,y,z,lw=1,c=color,zorder=99)
    fdat = np.array(fdat)
    cutoff_index = np.where(y > WL_SHORT.to_value(u.um))[0][0]

    s,c,f_rec = vpie.get_vpie(
            f_org=fdat,
            f_err=fdat*0.01,
            cutoff_index=cutoff_index,
            use_mean_error=True,
        )
    print(s)
    res = fdat - f_rec
    frac_res = res / fdat * 100
    for i, phi in enumerate(phase):
        color = cmap(phi/2/np.pi)
        x = np.ones_like(y) * phi
        z = np.log10(f_rec[i,:])
        ax1.plot(x,y,z,lw=1,c=color,ls='--',zorder=100)
        
    cmap = plt.cm.bwr
    yy,xx = np.meshgrid(y,phase)
    zz = np.full_like(xx,np.min(np.log10(fdat))-0.1)
    vminmax = np.max(np.abs(frac_res))
    norm = CenteredNorm(vcenter=0,halfrange=vminmax)
    colors = cmap(norm(frac_res))
    ax1.plot_surface(xx,yy,zz,facecolors=colors,shade=False,zorder=-100)
    im = ScalarMappable(norm, cmap)
    cbar = fig.colorbar(im, ax=ax1, orientation='horizontal', shrink=0.8, pad=0.05)
    cbar.set_label('% residual',fontsize=LG_FONTSIZE)

    
    ax1.zaxis.set_rotate_label(False)
    ax1.set_ylabel('NIR $\\leftarrow\\lambda\\rightarrow$ MIR',fontsize=FONTSIZE,labelpad=-10)
    ax1.set_zlabel('$\\mathbf{f}_{\\lambda}$\t',fontsize=FONTSIZE,labelpad=-10)
    ax1.set_xlabel('$t$',fontsize=FONTSIZE,labelpad=-10)
    ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE,length=0)
    ax1.set_xlim(0,2*np.pi)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.view_init(elev=20, azim=30)
    # ax1.view_init(elev=0, azim=30)
    ax1.spines[['right', 'top']].set_visible(False)
    
    
    
    
    fig.savefig(OUTPATH)
    plt.close(fig)