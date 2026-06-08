"""
3D figs to explain the VPIE method visually
"""
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
from common import figure_context, COLWIDTH

plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'


cmap = cm_teal_cream_orange
BASE_CFG = paths.static / 'base.cfg'
PREFIX = 'fig3d'

PL_MULTIPLIER = 10e3
FONTSIZE = 14
LG_FONTSIZE = 12
WL_SHORT = 1.5*u.um
FIGSIZE = (COLWIDTH, COLWIDTH)
RATIOS = [20, 1]

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
    with figure_context(figsize=FIGSIZE) as fig:
        gs = fig.add_gridspec(2, 1, height_ratios=RATIOS, hspace=0.1)
        ax: Axes3D = fig.add_subplot(gs[0, 0], projection='3d')
        cbar_ax = fig.add_subplot(gs[1, 0])
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
                               cfg.geometry.observer_altitude.value**2)\
                .to_value(u.dimensionless_unscaled)
            f_total = f_planet * PL_MULTIPLIER + f_star
            f_small = f_planet + f_star
            fdat_big_planet.append(f_total)
            fdat_real.append(f_small)
            z = np.log10(f_total)
            x = np.ones_like(y) * phi
            ax.plot(x, y, z, lw=2, c=color, zorder=100)

        ax.set_ylabel('SW $\\leftarrow\\lambda\\rightarrow$ LW',
                      fontsize=FONTSIZE, labelpad=-10)
        ax.set_zlabel(r'$\bm{f}$', fontsize=FONTSIZE, labelpad=-10)
        ax.set_xlabel('$t$', fontsize=FONTSIZE, labelpad=-10)
        ax.tick_params(axis='both', which='major',
                       labelsize=FONTSIZE, length=0)
        ax.zaxis.set_rotate_label(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=30)
        ax.spines[['right', 'top']].set_visible(False)
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
        ax.plot_surface(xx, yy, zz, facecolors=colors,
                        shade=False, zorder=-100)
        im = ScalarMappable(norm, dcmap)
        cbar = fig.colorbar(
            im, cax=cbar_ax, orientation='horizontal', shrink=0.8, pad=0.05)
        cbar.set_label('variation (\\%)', fontsize=LG_FONTSIZE)
        # End bottom part
        fig.savefig(paths.figures / f'{PREFIX}_a.pdf')
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        zlims = ax.get_zlim()

    cutoff_index = np.where(y > WL_SHORT.to_value(u.um))[0][0]

    s, c, f_rec = vpie.get_vpie(
        f_org=fdat_real,
        f_err=fdat_real*0.01,
        cutoff_index=cutoff_index,
        use_mean_error=True,
        ic_string='bic'
    )
    f_rec_real = vpie.get_reconstruction(fdat_real, c, s)
    print(s)

    with figure_context(figsize=(FIGSIZE[0], 0.9*FIGSIZE[1])) as fig:
        gs = fig.add_gridspec(1, 1)
        ax: Axes3D = fig.add_subplot(gs[0, 0], projection='3d')
        # cbar_ax = fig.add_subplot(gs[1, 0])
        # cbar_ax.set_visible(False)
        for i, phi in enumerate(phase):
            color = cmap(phi/2/np.pi)
            if i in s:
                x = np.ones_like(y) * phi
                z = np.log10(fdat_big_planet[i, :])
                ax.plot(x, y, z, lw=2, c=color)

        ax.zaxis.set_rotate_label(False)
        ax.set_ylabel('SW $\\leftarrow\\lambda\\rightarrow$ LW',
                      fontsize=FONTSIZE, labelpad=-10)
        ax.set_zlabel(r'$\bm{f}^{(k)}$', fontsize=FONTSIZE, labelpad=-10)
        ax.set_xlabel('$t$', fontsize=FONTSIZE, labelpad=-10)
        ax.tick_params(axis='both', which='major',
                       labelsize=FONTSIZE, length=0)
        ax.set_xlim(0, 2*np.pi)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=30)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)
        cbar = fig.colorbar(
            im, cax=cbar_ax, orientation='horizontal', shrink=0.8, pad=0.05)
        cbar.set_label('variation (\\%)', fontsize=LG_FONTSIZE)
        fig.savefig(paths.figures / f'{PREFIX}_b.pdf')

    with figure_context(figsize=FIGSIZE) as fig:
        gs = fig.add_gridspec(2, 1, height_ratios=RATIOS, hspace=0.1)
        ax: Axes3D = fig.add_subplot(gs[0, 0], projection='3d')
        cbar_ax = fig.add_subplot(gs[1, 0])
        for i, phi in enumerate(phase):
            color = cmap(phi/2/np.pi)
            x = np.ones_like(y) * phi
            z = np.log10(fdat_big_planet[i, :])
            ax.plot(x, y, z, lw=2, c=color, zorder=100)
            z = np.log10(f_rec[i, :])
            ax.plot(x, y, z, lw=2, c=color, zorder=100, ls='--')

        ax.view_init(elev=20, azim=30)

        ax.tick_params(axis='both', which='major',
                       labelsize=FONTSIZE, length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.zaxis.set_rotate_label(False)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_ylabel('SW $\\leftarrow\\lambda\\rightarrow$ LW',
                      fontsize=FONTSIZE, labelpad=-10)
        ax.set_zlabel(r'$\bm{f},\,\tilde{\bm{f}}$',
                      fontsize=FONTSIZE, labelpad=-10)
        ax.set_xlabel('$t$', fontsize=FONTSIZE, labelpad=-10)

        # Bottom part
        # pylint: disable-next=no-member
        dcmap = plt.cm.bwr
        yy, xx = np.meshgrid(y, phase)
        zz = np.full_like(xx, np.min(np.log10(fdat_big_planet))-0.1)
        res = fdat_real - f_rec_real
        frac_res = res / fdat_real * 1e6
        vminmax = np.max(np.abs(frac_res))
        norm = CenteredNorm(vcenter=0, halfrange=vminmax)
        colors = dcmap(norm(frac_res))
        ax.plot_surface(xx, yy, zz, facecolors=colors,
                        shade=False, zorder=-100)
        im = ScalarMappable(norm, dcmap)
        cbar = fig.colorbar(
            im, cax=cbar_ax, orientation='horizontal', shrink=0.8, pad=0.05)
        cbar.set_label('residual (ppm)', fontsize=LG_FONTSIZE)
        # End bottom part
        fig.savefig(paths.figures / f'{PREFIX}_c.pdf')
