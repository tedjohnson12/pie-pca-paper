"""
Affects of changing inclination on the dayside flux and day/night flux ratio
for GJ 876 d

Note that it makes the most sense to do this with a low thermal inertia parameter
since if they heat is already being transported efficiently, then the night/day
temperature ratio is already nearly 1 and there will be very little effect from
inclination.
"""

from pathlib import Path
from copy import deepcopy
import numpy as np
import VSPEC
from loguru import logger
import libpypsg as psg
from astropy import units as u

import gj876_run as gj876

EPSILON = 1e-2
INCLINATIONS = np.array([90,86, 60, 30,10,5])*u.deg


def get_parameters(
    inclination: u.Quantity,
    epsilon: float
)-> VSPEC.params.InternalParameters:
    _gcm = gj876.GCM_DICT.copy()
    _gcm['gcm']['vspec']['epsilon'] = epsilon
    _header = deepcopy(gj876.HEADER)
    _header.data_path = Path(__file__).parent / '.vspec' / f'gj876_inc{inclination.to_value(u.deg):.0f}'
    _system = deepcopy(gj876.SYSTEM)
    _system.inclination = inclination
    _psg = deepcopy(gj876.PSG)
    _psg.gcm_binning = 6
    return VSPEC.params.InternalParameters(
        header=_header,
        star=gj876.STAR,
        planet=gj876.PLANET,
        system=_system,
        obs=gj876.OBS,
        psg=_psg,
        inst=gj876.INST,
        gcm=VSPEC.params.gcmParameters.from_dict(
            _gcm
        )
    )

def main(rerun: bool):
    dat = {}
    for i in INCLINATIONS:
        params = get_parameters(i, EPSILON)
        model = VSPEC.ObservationModel(params)
        planet_is_built = (model.directories['psg_combined'] / 'phase00000.fits').exists()
        if rerun or not planet_is_built:
            logger.info(f'Running build_planet for {i.to_value(u.deg)}')
            model.build_planet()
        else:
            logger.info(f'Skipping build_planet ran for {i.to_value(u.deg)}')
        spectra_is_built = (model.directories['all_model'] / 'phase00000.fits').exists()
        if rerun or not spectra_is_built:
            logger.info(f'Running build_spectra for {i.to_value(u.deg)}')
            model.build_spectra()
        else:
            logger.info(f'Skipping build_spectra ran for {i.to_value(u.deg)}')
        
        data = VSPEC.PhaseAnalyzer.from_model(model)
        thermal = data.thermal
        wl = data.wavelength
        wl_to_use = 5*u.um
        wl_idx = np.argmin(np.abs(wl - wl_to_use))
        thermal_at_wl = thermal[wl_idx,:]
        thermal_min = thermal_at_wl.min()
        thermal_max = thermal_at_wl.max()
        logger.info(f'Minimum thermal flux at {wl_to_use:.2f} is {thermal_min:.2e}')
        logger.info(f'Maximum thermal flux at {wl_to_use:.2f} is {thermal_max:.2e}')
        dat[(i.to_value(u.deg))] = {
            'min': thermal_min,
            'max': thermal_max
        }
    return dat

    
    
if __name__ == '__main__':
    psg.docker.set_url_and_run()
    dat = main(True)
    for i in INCLINATIONS:
        _i = i.to_value(u.deg)
        print(f'$F_\\mathrm{{night}}/F_\\mathrm{{day}}(i={_i}^\\circ) = {dat[_i]["min"]/dat[_i]["max"]:.2e}$')
    for i in INCLINATIONS:
        _i = i.to_value(u.deg)
        print(f'$F_\\mathrm{{day}}(i={_i}^\\circ) / F_\\mathrm{{day}}(i=90^\\circ) = {dat[_i]["max"]/dat[90]["max"]:.2e}$')