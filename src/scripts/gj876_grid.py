"""
GJ 876 retrieval grid
"""

from pathlib import Path
import numpy as np
import VSPEC
import asdf
from loguru import logger
import libpypsg as psg
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

import paths
from gj876_run import get_grid_params
from common import get_temperature_ratio

GRID_DIR = paths.data / 'grid_gj876'

LOG_EPSILON_GRID = np.linspace(-3, 3, 25)
TRAT_GRID = [get_temperature_ratio(10**_log_epsilon)
             for _log_epsilon in LOG_EPSILON_GRID]

dt_to_eps = RegularGridInterpolator([TRAT_GRID,], LOG_EPSILON_GRID)


def _fpath(_log_epsilon) -> Path:
    return GRID_DIR / f'epsilon_{_log_epsilon:.2f}.asdf'


def _run(_log_epsilon: float):
    _path = _fpath(_log_epsilon)
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
        _path = _fpath(_log_epsilon)
        thermal = asdf.open(_path)['thermal'][:, :]
        vals.append(thermal)
    return RegularGridInterpolator([LOG_EPSILON_GRID,], np.array(vals))


if __name__ == '__main__':
    psg.docker.set_url_and_run()
    for log_epsilon in LOG_EPSILON_GRID:
        _run(log_epsilon)
    print(TRAT_GRID)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(LOG_EPSILON_GRID, TRAT_GRID, c='k')
    ax.set_xlabel('$\\log \\epsilon$')
    ax.set_ylabel('$T_{\\rm night} / T_{\\rm day}$')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    plt.savefig(paths.figures / 'temp_rat_relation.png')
