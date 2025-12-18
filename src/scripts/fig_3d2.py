import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy import units as u
from time import sleep
from matplotlib.cm import ScalarMappable
from matplotlib.colors import CenteredNorm

import libpypsg as psg
from GridPolator import GridSpectra
import vpie

from colors import cm_teal_cream_orange
import paths

plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'


cmap = cm_teal_cream_orange
BASE_CFG = paths.static / 'base.cfg'
OUTPATH = paths.figures / 'fig_3d2.pdf'

PL_MULTIPLIER = 1e4
FONTSIZE = 14
LG_FONTSIZE = 12
WL_SHORT = 1.5*u.um

TEFF_PHOT = np.array([3000])
TEFF_SPOT = np.array([2600])

TEFFS = [2600, 2700, 2800, 2900, 3000]

phase = np.linspace(0, 2*np.pi, 20)
spot_frac = 0.2 + 0.05 * np.cos(phase/2)
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

    fig = plt.figure(figsize=(3, 7))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 0.1, 1, 1, 0.1], hspace=0.1)
    ax1: Axes3D = fig.add_subplot(gs[0, 0], projection='3d')
    cbar_ax1 = fig.add_subplot(gs[1, 0])
    ax2: Axes3D = fig.add_subplot(gs[2, 0], projection='3d')
    ax3: Axes3D = fig.add_subplot(gs[3, 0], projection='3d')
    cbar_ax3 = fig.add_subplot(gs[4, 0])
    fdat_big_planet = []
    fdat_real = []

    for phi, fp, ts in zip(phase, spot_frac, tsurf):
        _cfg = cfg
        _cfg.surface.temperature = ts
        rad = psg.APICall(cfg, 'rad')().rad
        sleep(0.2)
        color = cmap(phi/2/np.pi)
        y = rad.wl.to_value(u.um)
        f_planet = rad['Thermal'].to_value(u.W/u.m**2/u.um)
        f_star = spec.evaluate([TEFF_SPOT])*fp + \
            spec.evaluate([TEFF_PHOT])*(1-fp)
        f_star = f_star[0]
        f_star = f_star * (cfg.target.star_radius.value**2 /
                           cfg.geometry.observer_altitude.value**2).to_value(u.dimensionless_unscaled)
        f_total = f_planet * PL_MULTIPLIER + f_star
        f_small = f_planet + f_star
        fdat_big_planet.append(f_total)
        fdat_real.append(f_small)
        z = np.log10(f_total)
        x = np.ones_like(y) * phi
        ax1.plot(x, y, z, lw=2, c=color, zorder=100)

    ax1.set_ylabel('NIR $\\leftarrow\\lambda\\rightarrow$ MIR',
                   fontsize=FONTSIZE, labelpad=-10)
    ax1.set_zlabel(r'$\bm{f}$', fontsize=FONTSIZE, labelpad=-10)
    ax1.set_xlabel('$t$', fontsize=FONTSIZE, labelpad=-10)
    ax1.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=0)
    ax1.zaxis.set_rotate_label(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])
    ax1.view_init(elev=20, azim=30)
    ax1.spines[['right', 'top']].set_visible(False)
    fdat_big_planet = np.array(fdat_big_planet)
    fdat_real = np.array(fdat_real)

    # Bottom part
    # pylint: disable-next=no-member
    dcmap = plt.cm.bwr
    yy, xx = np.meshgrid(y, phase)
    zz = np.full_like(xx, np.min(np.log10(fdat_big_planet))-0.1)
    mean_flux = np.mean(fdat_real, axis=0)
    rel_flux = (fdat_real - mean_flux) / mean_flux * 100
    vminmax = np.max(np.abs(rel_flux))
    norm = CenteredNorm(vcenter=0, halfrange=vminmax)
    colors = dcmap(norm(rel_flux))
    ax1.plot_surface(xx, yy, zz, facecolors=colors, shade=False, zorder=-100)
    im = ScalarMappable(norm, dcmap)
    cbar = fig.colorbar(im, cax=cbar_ax1, orientation='horizontal')
    cbar.set_label('variation (\\%)', fontsize=LG_FONTSIZE)
    # End bottom part

    cutoff_index = np.where(y > WL_SHORT.to_value(u.um))[0][0]

    s, c, f_rec = vpie.get_vpie(
        f_org=fdat_big_planet,
        f_err=fdat_big_planet*0.01,
        cutoff_index=cutoff_index,
        use_mean_error=True,
    )
    f_rec_real = vpie.get_reconstruction(fdat_real, c, s)
    print(s)
    for i, phi in enumerate(phase):
        color = cmap(phi/2/np.pi)
        if i in s:
            x = np.ones_like(y) * phi
            z = np.log10(fdat_big_planet[i, :])
            ax2.plot(x, y, z, lw=2, c=color)

    ax2.zaxis.set_rotate_label(False)
    ax2.set_ylabel('NIR $\\leftarrow\\lambda\\rightarrow$ MIR',
                   fontsize=FONTSIZE, labelpad=-10)
    ax2.set_zlabel(r'$\bm{f}^{(k)}$', fontsize=FONTSIZE, labelpad=-10)
    ax2.set_xlabel('$t$', fontsize=FONTSIZE, labelpad=-10)
    ax2.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=0)
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.view_init(elev=20, azim=30)
    ax2.spines[['right', 'top']].set_visible(False)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_zlim(ax1.get_zlim())

    for i, phi in enumerate(phase):
        color = cmap(phi/2/np.pi)
        x = np.ones_like(y) * phi
        z = np.log10(f_rec[i, :])
        ax3.plot(x, y, z, lw=2, c=color, zorder=100)

    ax3.view_init(elev=20, azim=30)

    ax3.tick_params(axis='both', which='major', labelsize=FONTSIZE, length=0)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.zaxis.set_rotate_label(False)
    ax3.spines[['right', 'top']].set_visible(False)
    ax3.set_ylabel('NIR $\\leftarrow\\lambda\\rightarrow$ MIR',
                   fontsize=FONTSIZE, labelpad=-10)
    ax3.set_zlabel(r'$\tilde{\bm{f}}$', fontsize=FONTSIZE, labelpad=-10)
    ax3.set_xlabel('$t$', fontsize=FONTSIZE, labelpad=-10)

    # Bottom part
    # pylint: disable-next=no-member
    dcmap = plt.cm.bwr
    yy, xx = np.meshgrid(y, phase)
    zz = np.full_like(xx, np.min(np.log10(fdat_big_planet))-0.1)
    mean_flux = np.mean(f_rec_real, axis=0)
    rel_flux = (f_rec_real - mean_flux) / mean_flux * 100
    vminmax = np.max(np.abs(rel_flux))
    norm = CenteredNorm(vcenter=0, halfrange=vminmax)
    colors = dcmap(norm(rel_flux))
    ax3.plot_surface(xx, yy, zz, facecolors=colors, shade=False, zorder=-100)
    im = ScalarMappable(norm, dcmap)
    cbar = fig.colorbar(im, cax=cbar_ax3, orientation='horizontal')
    cbar.set_label('variation (\\%)', fontsize=LG_FONTSIZE)
    # End bottom part

    fig.savefig(OUTPATH)
    plt.close(fig)
