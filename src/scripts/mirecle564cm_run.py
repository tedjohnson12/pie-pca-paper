"""
Simulate an observation of a hot sub-Neptune with JWST
"""

from pathlib import Path
from copy import deepcopy
import numpy as np
import astropy.units as u
import VSPEC
import libpypsg as psg

from proxb_run import (
    STAR,
    PLANET,
    SYSTEM,
    OBS,
    PSG,
    GCM_DICT,
    GCM
)


TRUE_EPSILON = 0.1
SHORT_WL_CUTOFF = 5*u.um
SW_MAX = 3*u.um
LW_MIN = 12.0*u.um
RERUN_PLANET = False
RERUN_SPECTRA = False


HEADER = VSPEC.params.Header(
    data_path=Path(__file__).parent / '.vspec' /
    f'proxb564cm_{TRUE_EPSILON:.2f}',
    seed=11,
    spec_grid=VSPEC.params.VSPECGridParameters(
        max_teff=3100 * u.K,
        min_teff=2600 * u.K,
        impl_bin='rust',
        impl_interp='scipy',
        fail_on_missing=False
    ),
    desc='Proxb with 6m MIRECLE',
)

# INST = VSPEC.params.InstrumentParameters.miri_lrs()
INST = VSPEC.params.InstrumentParameters(
    telescope=VSPEC.params.SingleDishParameters(
        aperture=5.64*u.m,
        zodi=1.0
    ),
    bandpass=VSPEC.params.BandpassParameters(
        wl_blue=1*u.um,
        wl_red=18*u.um,
        resolving_power=50,
        wavelength_unit=u.micron,
        flux_unit=u.Unit('W m-2 um-1')
    ),
    detector=VSPEC.params.DetectorParameters(
        beam_width=0.8*u.arcsec,
        integration_time=10*u.s,
        ccd=VSPEC.params.ccdParameters(
            pixel_sampling=1,
            read_noise=16.8*u.electron,
            dark_current=100*u.electron/u.s,
            throughput=0.7,
            emissivity=0.1,
            temperature=35*u.K
        )
    )
)



VSPEC_PARAMS = VSPEC.params.InternalParameters(
    header=HEADER,
    star=STAR,
    planet=PLANET,
    system=SYSTEM,
    obs=OBS,
    psg=PSG,
    inst=INST,
    gcm=GCM
)


def get_grid_params(epsilon: float):
    """
    Get the grid parameters with given epsilon
    """

    _gcm = GCM_DICT.copy()
    _gcm['gcm']['vspec']['epsilon'] = epsilon
    _header = deepcopy(HEADER)
    _header.data_path = Path(__file__).parent / '.vspec' / 'mirecle-grid'
    return VSPEC.params.InternalParameters(
        header=_header,
        star=STAR,
        planet=PLANET,
        system=SYSTEM,
        obs=OBS,
        psg=PSG,
        inst=INST,
        gcm=VSPEC.params.gcmParameters.from_dict(
            _gcm
        )
    )


def get_model():
    """
    Return the fiducial model
    """
    return VSPEC.ObservationModel(VSPEC_PARAMS)


def get_teq():
    """
    Compute the equilibrium temperature
    """
    return STAR.teff * np.sqrt(STAR.radius / PLANET.semimajor_axis/2)\
        * (1-GCM_DICT['gcm']['vspec']['albedo'])


if __name__ == '__main__':
    model = get_model()
    if not (model.directories['psg_thermal'] / 'phase00000.fits').exists() or RERUN_PLANET:
        psg.docker.set_url_and_run()
        model.build_planet()
    if not (model.directories['all_model'] / 'phase00000.fits').exists() or RERUN_SPECTRA:
        model.build_spectra()
