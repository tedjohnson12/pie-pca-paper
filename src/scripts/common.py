"""
Functions common to multiple scripts
"""
import contextlib
import matplotlib.pyplot as plt
from loguru import logger
from VSPEC.gcm.heat_transfer import get_equator_curve

import numpy as np

COLWIDTH = 8.5/2
FIGSIZE = (COLWIDTH, 0.7*COLWIDTH)


@contextlib.contextmanager
def figure_context(*args, **kwargs):
    """
    Automatically close the figure when you are done.
    This helps keep everything neat.
    """
    fig: plt.Figure = plt.figure(*args, **kwargs)
    yield fig
    plt.close(fig)


def bin_image(im: np.ndarray, nwl: int, ntime: int, power: int):
    """
    Reduce image size by a 2d window size

    :param im: The original image
    :type im: np.ndarray
    :param nwl: Wavelength window size
    :type nwl: int
    :param ntime: Time window size
    :type ntime: int
    :param power: The power to use in averaging. 1 for linear, 2 for quadrature, etc.
    :type power: int

    :return: The binned image
    """
    logger.warning(
        'Python function `bin_image` is deprecated. '
        'Use the rust implementation `vpie.bin_image` instead for 15x speed.')

    def add(*args):
        _sum = args[0] * 0
        for arg in args:
            _sum += arg**power
        return _sum**(1/power) / len(args)
    im = np.atleast_2d(im)
    original_size_time, original_size_wl = im.shape
    new_size_time = int(np.ceil(original_size_time/ntime))
    new_size_wl = int(np.ceil(original_size_wl/nwl))
    out_arr = np.zeros((new_size_time, new_size_wl))
    for i in range(new_size_time):
        for j in range(new_size_wl):
            sub_arr = (im[i*ntime:min((i+1)*ntime, original_size_time),
                       j*nwl:min((j+1)*nwl, original_size_wl)]).flatten()
            val = add(*sub_arr)
            out_arr[i, j] = val
    return out_arr


def fold_image(im: np.ndarray, stride: int, power: int):
    """
    Fold image in phase
    """
    logger.warning(
        'Python function `fold_image` is deprecated. '
        'Use the rust implementation `vpie.fold_image` instead for 15x speed.')
    im = np.atleast_2d(im)
    original_size_time, original_size_wl = im.shape
    size_stack = (original_size_time // stride) - 1
    if original_size_time % stride != 0:
        size_stack += 1
    out_arr = np.zeros((stride, original_size_wl, size_stack)) * np.nan
    for i in range(size_stack):
        for j in range(stride):
            for k in range(original_size_wl):
                val = im[i*stride+j, k]
                out_arr[j, k, i] = val
    denom = np.sum(~np.isnan(out_arr), axis=2)
    summed = np.nansum(out_arr**power, axis=2)**(1/power)/denom
    return summed


def find_eclipse(a: np.ndarray):
    """
    Find the start and end of an eclipse
    """
    _next = np.concatenate([a[1:], a[-1:]])
    diff = _next-a
    return np.argmin(diff)+1, np.argmax(diff)


def remove_epoch(thermal: np.ndarray, i_start: int, i_end: int):
    """
    Remove an epoch from an observation
    """
    if thermal.ndim == 1:
        return np.concatenate([thermal[:i_start], thermal[i_end+1:]], axis=0)
    return np.concatenate([thermal[:i_start, :], thermal[i_end+1:, :]], axis=0)


def cite(k: str):
    """
    Table citation for ref id `k`. E.g., `'moran2023'`-> `\\citet{moran2023}`
    """

    if k in ['assumed']:
        return k
    else:
        return f'\\citet{{{k}}}'


def foot(t):
    """
    LaTeX superscript
    """
    return rf'$^{t}$'


def get_temperature_ratio(epsilon: float):
    """
    Compute the night/day temperature ratio
    """
    if epsilon < 1:
        mode = 'ivp_reflect'
    elif epsilon < 10:
        mode = 'bvp'
    else:
        mode = 'analytic'
    _, tsurf = get_equator_curve(epsilon, 180, mode)
    return np.min(tsurf)/np.max(tsurf)
