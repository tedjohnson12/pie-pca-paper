"""
1D JWST retrieval grid
"""

from pathlib import Path
import numpy as np
import VSPEC
import asdf
from loguru import logger
import libpypsg as psg
from scipy.interpolate import RegularGridInterpolator

import paths
from mirecle564cm_run import get_grid_params
from common import get_temperature_ratio

GRID_DIR = paths.data / 'grid_mirecle564cm'

LOG_EPSILON_GRID = np.linspace(-3, 3, 25)
TRAT_GRID = [get_temperature_ratio(10**_log_epsilon)
             for _log_epsilon in LOG_EPSILON_GRID]

dt_to_eps = RegularGridInterpolator([TRAT_GRID,], LOG_EPSILON_GRID)


def fpath(_log_epsilon) -> Path:
    """
    Path to a grid file
    """
    return GRID_DIR / f'epsilon_{_log_epsilon:.2f}.asdf'


def run(_log_epsilon: float):
    """
    Run a model and store it on disk
    """
    _path = fpath(_log_epsilon)
    if _path.exists():
        return None
    logger.info(f'Running log epsilon: {_log_epsilon}')
    params = get_grid_params(10**_log_epsilon)
    model = VSPEC.ObservationModel(params)
    model.build_planet()
    model.build_spectra()
    data = VSPEC.PhaseAnalyzer.from_model(model)
    thermal = data.thermal
    tree = {
        'epsilon': _log_epsilon,
        'wavelength': data.wavelength,
        'time': data.time,
        'thermal': thermal
    }
    af = asdf.AsdfFile(tree)

    _path.parent.mkdir(parents=True, exist_ok=True)
    with open(_path, 'wb') as f:
        af.write_to(f)


def get_interp() -> RegularGridInterpolator:
    """
    Get interpolator
    """
    vals = []
    for _log_epsilon in LOG_EPSILON_GRID:
        _path = fpath(_log_epsilon)
        thermal = asdf.open(_path)['thermal'][:, :]
        vals.append(thermal)
    return RegularGridInterpolator([LOG_EPSILON_GRID,], np.array(vals))


if __name__ == '__main__':
    if not GRID_DIR.exists():
        GRID_DIR.mkdir(parents=True, exist_ok=True)
    psg.docker.set_url_and_run()
    for log_epsilon in LOG_EPSILON_GRID:
        run(log_epsilon)
