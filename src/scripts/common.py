
import contextlib
import matplotlib.pyplot as plt

import numpy as np


COLWIDTH = 8.5/2
FIGSIZE = (COLWIDTH, 0.7*COLWIDTH)


@contextlib.contextmanager
def figure_context(*args, **kwargs):
    fig: plt.Figure = plt.figure(*args, **kwargs)
    yield fig
    plt.close(fig)


def bin_image(im: np.ndarray, nwl:int, ntime: int, power: int):
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
            sub_arr = (im[i*ntime:min((i+1)*ntime,original_size_time), j*nwl:min((j+1)*nwl,original_size_wl)]).flatten()
            val = add(*sub_arr)
            out_arr[i,j] = val
    return out_arr

def find_eclipse(a: np.ndarray):
    next = np.concatenate([a[1:], a[-1:]])
    diff = next-a
    return np.argmin(diff)+1, np.argmax(diff)

def remove_epoch(thermal:np.ndarray, i_start: int, i_end: int):
    if thermal.ndim == 1:
        return np.concatenate([thermal[:i_start], thermal[i_end+1:]], axis=0)
    return np.concatenate([thermal[:i_start,:], thermal[i_end+1:,:]], axis=0)