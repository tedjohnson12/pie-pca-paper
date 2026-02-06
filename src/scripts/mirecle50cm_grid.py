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
from mirecle50cm_run import get_grid_params, get_temperature_ratio

GRID_DIR = paths.data / 'grid_mirecle50cm'

LOG_EPSILON_GRID = np.linspace(-3, 3, 25)
TRAT_GRID = [get_temperature_ratio(10**log_epsilon)
             for log_epsilon in LOG_EPSILON_GRID]

dt_to_eps = RegularGridInterpolator([TRAT_GRID,], LOG_EPSILON_GRID)


def fpath(log_epsilon) -> Path:
    return GRID_DIR / f'epsilon_{log_epsilon:.2f}.asdf'


def run(log_epsilon: float):
    _path = fpath(log_epsilon)
    if _path.exists():
        return None
    logger.info(f'Running log epsilon: {log_epsilon}')
    params = get_grid_params(10**log_epsilon)
    model = VSPEC.ObservationModel(params)
    model.build_planet()
    model.build_spectra()
    data = VSPEC.PhaseAnalyzer.from_model(model)
    thermal = data.thermal
    tree = {
        'epsilon': log_epsilon,
        'wavelength': data.wavelength,
        'time': data.time,
        'thermal': thermal
    }
    af = asdf.AsdfFile(tree)

    _path.parent.mkdir(parents=True, exist_ok=True)
    with open(_path, 'wb') as f:
        af.write_to(f)
    del af, tree, thermal, data, model, params


def get_interp() -> RegularGridInterpolator:
    vals = []
    for log_epsilon in LOG_EPSILON_GRID:
        _path = fpath(log_epsilon)
        thermal = asdf.open(_path)['thermal'][:, :]
        vals.append(thermal)
    return RegularGridInterpolator([LOG_EPSILON_GRID,], np.array(vals))


if __name__ == '__main__':
    if not GRID_DIR.exists():
        GRID_DIR.mkdir(parents=True, exist_ok=True)
    psg.docker.set_url_and_run()
    for log_epsilon in LOG_EPSILON_GRID:
        run(log_epsilon)
