"""
Script that querys the NExScI database
and creates a figure based on the data.

Created by Ted Johnson (NASA GSFC 693) in Oct 2022,
uploaded to Github 2023-04-07

"""
from pathlib import Path
from typing import Callable
from datetime import datetime
from io import StringIO
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import requests
from astropy import units as u, constants as c

import paths

OUTPATH = paths.figures / 'nearby_mdwarfs.pdf'
DATEPATH = paths.output / 'nearby_mdwarfs_date.txt'
BIBPATH = paths.tex / 'nea.bib'

TRANSIT_COLOR = 'DE5126'
NONTRANSIT_COLOR = '458977'
LINE_COLOR = '#9E5842'

CREDIT_TEXT_SIZE = 12
SOLAR_SYSTEM_FONT_SIZE = 16
AXES_LABEL_FONT_SIZE = 16
TICK_LABEL_SIZE = 16
LEGEND_TEXT_SIZE = 12
FIGSIZE = (10, 6)


def build_query(max_teff: int, max_dist: float) -> str:
    """
    Build the query to send to NExScI.

    Parameters
    ----------
    max_teff : int
        The maximum effective temperature in K
    max_dist : float
        The maximum distance in parsecs

    Returns
    -------
    str
        The query string
    """
    q = f'select * from pscomppars where st_teff < {max_teff} and sy_dist < {max_dist}'
    query = '+'.join(q.split(' '))+'&format=csv'
    return query


def get_data(max_teff: int, max_dist: float) -> pd.DataFrame:
    """
    Get the data from NExScI.

    Parameters
    ----------
    max_teff : int
        The maximum effective temperature in K
    max_dist : float
        The maximum distance in parsecs

    Returns
    -------
    pd.DataFrame
        The data
    """
    query = build_query(max_teff, max_dist)
    url = f'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}'
    print(f'Querying {url}')
    response = requests.get(url, timeout=30)
    text = StringIO(response.text)
    return pd.read_csv(text)


def apply_corrections(
    _df: pd.DataFrame,
    max_period: float,
    max_mass: float,
    max_insolation: float
) -> pd.DataFrame:
    """
    Apply corrections to the data.

    Parameters
    ----------
    _df : pd.DataFrame
        The data
    max_period : float
        The maximum orbital period
    max_mass : float
        The maximum planet mass
    max_insolation : float
        The maximum insolation

    Returns
    -------
    pd.DataFrame
        The corrected data
    """
    # GJ 667 C radius
    is_gj667c = _df['hostname'] == 'GJ 667 C'
    _df.loc[is_gj667c, 'st_rad'] = 0.42
    _df['pl_approx_insol'] = 10**(_df['st_lum']) / (_df['pl_orbsmax'])**2
    t_eq_earth = 255
    _df['pl_eqt'] = t_eq_earth * _df['pl_approx_insol']**(1/4)

    dist_ratio = _df['pl_orbsmax']**-2
    lum_ratio = 10**_df['st_lum']
    _df['xuv_ratio'] = dist_ratio.values * lum_ratio.values**0.4

    log_xuv = np.log10(_df['xuv_ratio'].values) - np.log10(0.4)
    # pylint: disable-next=no-member
    escape_vel = np.sqrt(2*c.G * _df['pl_bmasse'].values*u.M_earth /
                         get_radius(_df['pl_bmasse'].values*u.M_earth, 1.0))
    log_vesc = np.log10(escape_vel.to_value(u.km/u.s)) - np.log10(5)
    dist_above_shoreline = log_xuv - log_vesc**4
    priority_metric = np.log(80) - np.log(13)

    _df['priority_metric'] = dist_above_shoreline/priority_metric
    print(_df['priority_metric'])

    has_all = ~np.isnan(_df['sy_dist']) & ~np.isnan(
        _df['pl_bmasse']) & ~np.isnan(_df['pl_orbper'])
    keep = (_df['pl_orbper'] < max_period) & has_all & (
        _df['pl_bmasse'] < max_mass) & (_df['pl_approx_insol'] < max_insolation)
    return _df[keep]


def get_radius(mass: u.Quantity, scale: float) -> u.Quantity:
    """
    M-R relation from 2023arXiv231016733E

    Parameters
    ----------
    mass : u.Quantity
        The mass of the planet.
    scale : float
        The scale factor.

    Returns
    -------
    u.Quantity
        The radius of the planet.
    """
    is_array = not mass.isscalar
    if not is_array:
        mass = np.array([mass.value]) * mass.unit
    k = np.zeros_like(mass.value) + 13.0
    beta = np.zeros_like(mass.value) + 0.012

    k = np.where(mass < 115*u.M_earth, 0.53, k)
    beta = np.where(mass < 115*u.M_earth, 0.68, beta)

    k = np.where(mass < 4.95*u.M_earth, 1.01, k)
    beta = np.where(mass < 4.95*u.M_earth, 0.28, beta)

    radius = mass.to_value(u.M_earth)**beta * k * u.R_earth * scale

    if not is_array:
        return radius[0]
    else:
        return radius


def get_mirecle_targets() -> pd.DataFrame:
    """
    Get the MIRECLE target list.

    Returns
    -------
    pd.DataFrame
        The MIRECLE target list
    """
    path = Path(__file__).parent / 'mirecle_targets.txt'
    mirecle_list = pd.read_csv(path, names=['name'])
    return mirecle_list


def get_hwo_targets() -> pd.DataFrame:
    """
    Get the HWO target list from Table A
    of https://exoplanetarchive.ipac.caltech.edu/docs/'
    '2645_NASA_ExEP_Target_List_HWO_Documentation_2023.pdf

    Returns
    -------
    pd.DataFrame
        The HWO target list.
    """
    path = Path(__file__).parent / 'hwo_targets.txt'
    hwo_list = pd.read_csv(path, engine='python')
    return hwo_list


def print_demographics(_df):
    """
    Print some demographics for Ravi.

    Parameters
    ----------
    _df : pd.DataFrame
        The data
    """
    is_transit = _df['tran_flag'].values.astype('bool')
    n_transit = np.sum(is_transit)
    n_nontransit = np.sum(~is_transit)
    teq_max = _df['pl_eqt'].max()
    print(
        f'There are {n_transit} transiting exoplanets and '
        f'{n_nontransit} non-transiting exoplanets. The maximum '
        f'equilibrium temperature is {teq_max:.1f} K.')

def _write_citation(date: str):
    with open(BIBPATH, 'w', encoding='utf-8') as f:
        s = """@misc{nea13,
doi = {10.26133/NEA13},
url = {https://catcopy.ipac.caltech.edu/dois/doi.php?id=10.26133/NEA13},
author = {{NASA Exoplanet Science Institute}},
title = {Planetary Systems Composite Parameters Table},
version = """ + f"{{Last Accessed: {date}}}" + """,
publisher = {IPAC},
year = {2020} }"""
        f.write(s)


def setup_fig(credit=True):
    """
    Set up the figure.
    """
    plt.style.use('bmh')
    rc('font', weight='bold')
    _fig, _axes, = plt.subplots(1, 2, figsize=FIGSIZE, width_ratios=[30, 1])
    _fig.subplots_adjust(wspace=0.0)
    _ax = _axes[0]
    _cbarax = _axes[1]
    _ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

    date = datetime.now().strftime("%Y-%m-%d")
    dist_from_bottom = 0.05
    # _fig.text(0.4, dist_from_bottom, 'NASA Exoplanet Archive ' +
    #           date, fontfamily='serif', fontsize=CREDIT_TEXT_SIZE, weight='normal',ha='right')
    with open(DATEPATH, 'w', encoding='utf-8') as f:
        f.write(date)
    _write_citation(date)
    if credit:
        _fig.text(0.6, dist_from_bottom, 'Created by: Ted Johnson (UNLV, GSFC)',
                  fontfamily='serif', fontsize=CREDIT_TEXT_SIZE, ha='left', weight='normal')

    return _fig, _ax, _cbarax


def get_data_dicts(
    _df: pd.DataFrame,
    plot_mirecle: bool,
    plot_hwo: bool,
    size_func: Callable,
    _alpha: float,
    method: str
):
    """
    Get data for plotting.
    """
    if plot_hwo and plot_mirecle:
        raise ValueError('Please only choose one of HWO or MIRECLE.')
    is_transit = _df['tran_flag'].values.astype('bool')
    mirecle_target_list = get_mirecle_targets()
    hwo_target_list = get_hwo_targets()
    in_mirecle = _df.loc[:, 'pl_name'].isin(mirecle_target_list['name'])
    in_hwo = (
        _df.loc[:, 'hip_name'].isin(hwo_target_list['ID(HIP)'])
        | _df.loc[:, 'hd_name'].isin(hwo_target_list['ID(HD)'])
        | _df.loc[:, 'hostname'].isin(hwo_target_list['Common Name'])
    )
    x_kw = 'sy_dist'
    y_kw = 'pl_approx_insol'
    c_kw = 'st_teff'
    size_kw = 'pl_bmasse'
    data = []

    if method == 'transit':
        data.append(
            {
                'x': _df.loc[~is_transit, x_kw],
                'y': (_df.loc[~is_transit, y_kw]),
                'label': 'Non-transiting',
                'c': f'#{NONTRANSIT_COLOR}',
                's': size_func(_df.loc[~is_transit, size_kw]),
                'alpha': _alpha
            })
        data.append({
            'x': _df.loc[is_transit, x_kw],
            'y': (_df.loc[is_transit, y_kw]),
            'label': 'Transiting',
            'c': f'#{TRANSIT_COLOR}',
            's': size_func(_df.loc[is_transit, size_kw]),
            'alpha': _alpha
        })
    elif method == 'teff':
        data.append({
            'x': _df.loc[:, x_kw],
            'y': _df.loc[:, y_kw],
            'c': _df.loc[:, c_kw],
            's': size_func(_df.loc[:, size_kw]),
            'alpha': _alpha,
            'cmap': 'gist_heat',
            'edgecolors': 'k'
        })
    else:
        raise ValueError('Please choose a valid method.')

    if plot_mirecle:
        data.append({
            'x': _df.loc[in_mirecle, x_kw],
            'y': (_df.loc[in_mirecle, y_kw]),
            'label': 'MIRECLE Targets',
            'facecolors': 'none',
            'edgecolors': 'k',
            'linewidth': 2,
            's': size_func(_df.loc[in_mirecle, size_kw]),
            'alpha': _alpha
        })

    if plot_hwo:
        data.append({
            'x': _df.loc[in_hwo, x_kw],
            'y': (_df.loc[in_hwo, y_kw]),
            'label': 'HWO Targets',
            'facecolors': 'none',
            'edgecolors': 'k',
            'linewidth': 2,
            's': size_func(_df.loc[in_hwo, size_kw]),
            'alpha': _alpha
        })
    data.append({
        'x': np.nan,
        'y': -1,
        'label': 'Mars-mass',
        'c': 'k',
        's': size_func(0.107),
        'alpha': _alpha
    })
    data.append({
        'x': np.nan,
        'y': -1,
        'label': 'Earth-mass',
        'c': 'k',
        's': size_func(1),
        'alpha': _alpha
    })
    data.append({
        'x': np.nan,
        'y': -1,
        'label': 'Neptune-mass',
        'c': 'k',
        's': size_func(17.15),
        'alpha': _alpha
    })
    return data


def plot(
    _df: pd.DataFrame,
    plot_mirecle: bool,
    plot_hwo: bool,
    size_func: Callable,
    _alpha: float,
    method: str,
    max_dist: float,
    credit: bool,
    output: str
):
    """
    Plot the data.

    Parameters
    ----------
    _df : pd.DataFrame
        The data
    _ax : plt.Axes
        The axes to plot on
    plot_mirecle : bool
        Whether to plot the MIRECLE targets
    plot_hwo : bool
        Whether to plot the HWO target list
    size_func : function
        The function to determine the marker size
    _alpha : float
        The marker transparency
    """
    data = get_data_dicts(_df, plot_mirecle, plot_hwo,
                          size_func, _alpha, method)

    fig, _ax, _cbar_ax = setup_fig(credit)
    fig: plt.Figure
    _ax: plt.Axes
    _cbar_ax: plt.Axes
    turn_off_cbar = True
    for d in data:
        _im = _ax.scatter(**d)
        if 'cmap' in d:
            _ax.get_figure().colorbar(_im, ax=_ax, cax=_cbar_ax,
                                      label='Stellar Effective Temperature (K)',
                                      pad=0.01, fraction=0.05, shrink=0.5)
            turn_off_cbar = False
    if turn_off_cbar:
        _cbar_ax.set_axis_off()

    add_solar_system_planets(_ax)
    add_legend(_ax, size_func, method)
    add_labels(_ax, max_dist)
    # fig.tight_layout()
    fig.savefig(output, dpi=300)


def add_solar_system_planets(_ax: plt.Axes):
    """
    Add the solar system planets to the plot.

    Parameters
    ----------
    _ax : plt.Axes
        The axes to plot on
    """
    xlo, xhi = _ax.get_xlim()
    lw = 1
    ls = (0, (5, 10))
    stop = 1
    x = xlo + 0.01*(xhi-xlo)
    text_height_scale = 0.8
    ha = 'left'
    rot = 0
    fontsize = SOLAR_SYSTEM_FONT_SIZE
    _alpha = 1
    color = LINE_COLOR
    zorder = -100

    a_venus = 0.723  # AU
    _ax.axhline(1/a_venus**2, 0, stop, c=color, lw=lw,
                alpha=_alpha, ls=ls, zorder=zorder)
    _ax.text(x, text_height_scale*1/a_venus**2, 'Venus', ha=ha, rotation=rot,
             fontfamily='serif', fontsize=fontsize, alpha=_alpha, weight='bold')

    a_mars = 1.523  # AU
    _ax.axhline(1/a_mars**2, 0, stop, c=color, lw=lw,
                alpha=_alpha, ls=ls, zorder=zorder)
    _ax.text(x, text_height_scale*1/a_mars**2, 'Mars', ha=ha, rotation=rot,
             fontfamily='serif', fontsize=fontsize, alpha=_alpha, weight='bold')

    a_mercury = 0.387  # AU
    _ax.axhline(1/a_mercury**2, 0, stop, c=color, lw=lw,
                alpha=_alpha, ls=ls, zorder=zorder)
    _ax.text(x, text_height_scale*1/a_mercury**2, 'Mercury', ha=ha, rotation=rot,
             fontfamily='serif', fontsize=fontsize, alpha=_alpha, weight='bold')

    _ax.set_yscale('log')


def add_legend(_ax: plt.Axes, size_func: Callable, method: str):
    """
    Add the legend.

    Parameters
    ----------
    _ax : plt.Axes
        The axes to plot on
    size_func : function
        The function to determine the marker size
    """
    lgnd = _ax.legend(prop={'size': LEGEND_TEXT_SIZE, 'family': 'serif'},
                      framealpha=0.7, loc='lower right')
    legend_marker_size = size_func(2)
    if method == 'transit':
        # pylint: disable-next=protected-access
        lgnd.legend_handles[0]._sizes = [legend_marker_size]
        # pylint: disable-next=protected-access
        lgnd.legend_handles[1]._sizes = [legend_marker_size]


def add_labels(_ax: plt.Axes, max_dist: float):
    """
    Add the labels.

    Parameters
    ----------
    _ax : plt.Axes
        The axes to plot on
    max_dist : float
    """
    _ax.set_xlabel('Distance (pc)', fontfamily='serif',
                   fontsize=AXES_LABEL_FONT_SIZE, fontweight='bold')
    _ax.set_ylabel('Stellar Insolation Flux\n(relative to Earth)',
                   fontfamily='serif', fontsize=AXES_LABEL_FONT_SIZE, fontweight='bold')
    _ax.set_title(f'Potentially Rocky Planets Within {max_dist:g} pc',
                  fontsize=AXES_LABEL_FONT_SIZE, fontfamily='serif', fontweight='bold')


if __name__ in '__main__':
    parser = argparse.ArgumentParser(
        prog='python make_figure.py',
        description='Creates a figure of nearby exoplanets.',
        epilog='Created by: Ted Johnson (GSFC 693) in Oct 2022, uploaded to Github 2023-04-07'
    )

    parser.add_argument('-t', '--max_teff', type=int,
                        default=4000, help='Maximum effective temperature in K')
    parser.add_argument('-d', '--max_dist', type=float,
                        default=20, help='Maximum distance in parsecs')
    parser.add_argument('-o', '--output', type=str,
                        default=OUTPATH, help='Output filename')
    parser.add_argument('-p', '--max_period', type=float,
                        default=25, help='Maximum orbital period in days')
    parser.add_argument('-m', '--max_mass', type=float,
                        default=20, help='Maximum planet mass in Earth masses')
    parser.add_argument('-i', '--max_insol', type=float, default=100,
                        help='Maximum planet insolation in Earth fluxes')
    parser.add_argument('-s', '--size', type=float,
                        default=1, help='Marker size scale factor')
    parser.add_argument('-a', '--alpha', type=float,
                        default=0.5, help='Transparency')
    parser.add_argument('--mirecle', action='store_true',
                        help='Include MIRECLE target list')
    parser.add_argument('--hwo', action='store_true',
                        help='Include HWO target list')
    parser.add_argument('--method', type=str,
                        default='transit',
                        help='Method to choose colors. Can be `transit` or `teff`'
                        )
    parser.add_argument('--no_credit', action='store_true',
                        help='Do not include credit in the figure. '
                        'Only Ted is allowed to use this one.')

    args = parser.parse_args()

    df = get_data(args.max_teff, args.max_dist)

    if len(df) == 0:
        S = f"""There were no exoplanets found with the following parameters:
        Maximum stellar effective temperature: {args.max_teff} K
        Maximum distance: {args.max_dist} pc
        Maximum orbital period: {args.max_period} days
        Maximum planet mass: {args.max_mass} Earth masses
        Maximum planet insolation: {args.max_insol} Earth fluxes
        """
        raise RuntimeError(S)

    df = apply_corrections(
        df,
        args.max_period,
        args.max_mass,
        args.max_insol
    )

    print_demographics(df)

    interesting_cols = ['pl_name', 'st_teff', 'pl_approx_insol', 'pl_eqt',
                        'tran_flag', 'pl_orbper', 'pl_bmasse', 'sy_dist',
                        'xuv_ratio', 'priority_metric']
    print(df[interesting_cols])
    # df.to_csv('nearby_exoplanets.csv')
    # df[interesting_cols].to_csv('nearby_exoplanets_info.csv')

    alpha = args.alpha  # transparency

    def size(mass: float):
        """
        Determine marker size based on the planet's mass

        Parameters
        ----------
        mass : float
            The planet's mass relative to Earth.

        Returns
        -------
        float
            The marker size.
        """
        k = 30*args.size  # scales size of markers
        return k*mass

    plot(df, args.mirecle, args.hwo, size, alpha, args.method,
         args.max_dist, args.no_credit, args.output)

    # ticks = [0.1,0.5,1,2,5,10,100]
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(ticks)
