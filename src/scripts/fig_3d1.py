from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from time import sleep

import libpypsg as psg
from GridPolator import GridSpectra

from colors import cm_teal_cream_orange
import paths

cmap = cm_teal_cream_orange

BASE_CFG = paths.static / 'base.cfg'
OUTPATH = paths.figures / 'fig_3d1.pdf'

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
    
    fig = plt.figure(figsize=(3, 3))
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
        ax1.plot(x,y,z,lw=2,c=color)
    ax1.zaxis.set_rotate_label(False)
    ax1.set_ylabel('NIR $\\leftarrow\\lambda\\rightarrow$ MIR',fontsize=FONTSIZE,labelpad=-10)
    ax1.set_zlabel('$\\mathbf{f}_{\\lambda}$\t',fontsize=FONTSIZE,rotation='vertical',labelpad=-10)
    ax1.set_xlabel('$t$',fontsize=FONTSIZE,labelpad=-10)
    ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE,length=0)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.view_init(elev=20, azim=30)
    ax1.spines[['right', 'top']].set_visible(False)
    
    fig.savefig(OUTPATH)
    plt.close(fig)