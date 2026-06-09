"""
Simulate an observation of TOI 519b with JWST
"""

from pathlib import Path
from copy import deepcopy
import numpy as np
import astropy.units as u
import VSPEC
import VSPEC.config
import VSPEC.gcm
import libpypsg as psg
import paths
from common import foot

TABLE_FILE = paths.output / 'toi519.txt'
FIDUCIAL_EPSILON = 1.3547855

TEFF = 3354
SPOT_FRAC = 0.1
TSPOT = 2700
TPHOT = ((TEFF**4 - SPOT_FRAC*TSPOT**4)/(1-SPOT_FRAC))**0.25
SW_MAX = 2.5*u.um
LW_MIN = 4*u.um
WL_BIN_SIZE = 6
TIME_BIN_SIZE = 4

RADIUS_SCALE_MIN = 0.1
RADIUS_SCALE_MAX = 2.0
TEMP_RATIO_MIN = 0.05
TEMP_RATIO_MAX = 0.99

RERUN_PLANET = False
RERUN_SPECTRA = False


HEADER = VSPEC.params.Header(
    data_path=Path(__file__).parent / '.vspec' /
    f'toi519_{FIDUCIAL_EPSILON:.2f}',
    seed=519,
    spec_grid=VSPEC.params.VSPECGridParameters(
        max_teff=3400 * u.K,
        min_teff=2600 * u.K,
        impl_bin='rust',
        impl_interp='scipy',
        fail_on_missing=False
    ),
    log_level='info',
    desc='TOI 519 b with JWST',
)

STAR = VSPEC.params.StarParameters(
    psg_star_template='M',
    teff=TPHOT * u.K,
    radius=0.3578 * u.R_sun,
    period=4 * u.day,
    misalignment=0 * u.deg,
    misalignment_dir=0 * u.deg,
    mass=0.37 * u.Msun,
    ld=VSPEC.params.LimbDarkeningParameters.proxima(),
    spots=VSPEC.params.SpotParameters(
        distribution='iso',
        initial_coverage=SPOT_FRAC,
        equillibrium_coverage=SPOT_FRAC,
        area_mean=500*VSPEC.config.MSH,
        area_logsigma=0.2,
        teff_umbra=TSPOT * u.K,
        teff_penumbra=TSPOT * u.K,
        burn_in=0 * u.day,
        growth_rate=0 * u.day**-1,
        decay_rate=0*VSPEC.config.MSH * u.day**-1,
        initial_area=10*VSPEC.config.MSH,
    ),
    faculae=VSPEC.params.FaculaParameters.none(),
    flares=VSPEC.params.FlareParameters.none(),
    granulation=VSPEC.params.GranulationParameters.none(),
    grid_params=1000
)

PLANET = VSPEC.params.PlanetParameters(
    name='toi519',
    radius=1.03*u.R_jup,
    gravity=VSPEC.params.GravityParameters(
        mode='kg',
        value=0.463*u.M_jup
    ),
    semimajor_axis=0.0159*u.AU,
    orbit_period=1.2652328*u.day,
    rotation_period=1.2652328*u.day,
    eccentricity=0.0,
    obliquity=0.0*u.deg,
    obliquity_direction=0.0*u.deg,
    init_substellar_lon=0.0*u.deg,
    init_phase=90*u.deg,
)

SYSTEM = VSPEC.params.SystemParameters(
    distance=115*u.pc,
    inclination=88.9*u.deg,
    phase_of_periastron=0*u.deg
)

OBS = VSPEC.params.ObservationParameters(
    observation_time=1.265*u.day,
    integration_time=30*u.min
)

PSG = VSPEC.params.psgParameters(
    gcm_binning=200,
    phase_binning=1,
    use_continuum_stellar=True,
    use_molecular_signatures=True,
    nmax=1,
    lmax=2,
    continuum=['Rayleigh', 'CIA_all', 'Refraction']
)

INST = VSPEC.params.InstrumentParameters(
    telescope=VSPEC.params.SingleDishParameters.jwst(),
    bandpass=VSPEC.params.BandpassParameters(
        wl_blue=0.6*u.um,
        wl_red=5.0*u.um,
        resolving_power=100,
        wavelength_unit=u.micron,
        flux_unit=u.Unit('W m-2 um-1')
    ),
    detector=VSPEC.params.DetectorParameters(
        beam_width=0.5*u.arcsec,
        integration_time=10*u.s,
        ccd=VSPEC.params.ccdParameters(
            pixel_sampling=8,
            read_noise=16.8*u.electron,
            dark_current=0.005*u.electron/u.s,
            throughput=0.4,
            emissivity=0.1,
            temperature=50*u.K
        )
    )
)
GCM_DICT = {
    'star': {
        'teff': TEFF*u.K,
        'radius': STAR.radius
    },
    'planet': {
        'semimajor_axis': PLANET.semimajor_axis,
    },
    'gcm': {
        'mean_molec_weight': 2.0,
        'vspec': {
            'nlayer': 30,
            'nlon': 90,
            'nlat': 45,
            'epsilon': FIDUCIAL_EPSILON,
            'psurf': 2*u.bar,
            'ptop': 1e-5*u.bar,
            'wind': {'U': 0*u.m/u.s, 'V': 0*u.m/u.s},
            'gamma': 1.,
            'albedo': 0.0,
            'emissivity': 1.0,
            'molecules': {
                'N2': 1e-20,
            },
            'lat_redistribution': 0.0
        }
    }
}
GCM = VSPEC.params.gcmParameters.from_dict(
    GCM_DICT
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
    Get VSPEC parameters with given epsilon
    """
    _gcm = GCM_DICT.copy()
    _gcm['gcm']['vspec']['epsilon'] = epsilon
    _header = deepcopy(HEADER)
    _header.data_path = Path(__file__).parent / '.vspec' / 'toi519-grid'
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


def get_teq() -> u.Quantity:
    """
    Compute the equilibrium temperature
    """
    return TEFF*u.K * np.sqrt(STAR.radius / PLANET.semimajor_axis/2) \
        * (1-GCM_DICT['gcm']['vspec']['albedo'])


REF = {
    'assumed': '\\dagger',
    'kagetani2023': 'a',
    'gaiacollaboration2020': 'e'

}

TAB = {
    'Stellar Effective Temperature': f'{(TEFF*u.K):latex}{foot(REF["kagetani2023"])}',
    'Stellar Radius': f'{round(STAR.radius,2):latex}{foot(REF["kagetani2023"])}',
    'Stellar Rotation Period': f'{STAR.period:latex}{foot(REF["assumed"])}',
    'Photosphere Temperature': f'{STAR.teff.round(0):latex}',
    'Spot Temperature': f'{STAR.spots.teff_umbra:latex}{foot(REF["assumed"])}',
    'Spot Coverage Fraction': f'{SPOT_FRAC:.1f}{foot(REF["assumed"])}',
    'Planet Radius': f'{PLANET.radius.round(2):latex}{foot(REF["kagetani2023"])}',
    'Planet Mass': \
        f'{PLANET.gravity.value.to(u.M_earth).round(0):latex}{foot(REF["kagetani2023"])}',
    'Planet $T_\\mathrm{eq}$': f'{get_teq().to(u.K).round(0):latex}',
    'Semimajor Axis': f'{PLANET.semimajor_axis:latex}{foot(REF["kagetani2023"])}',
    'Orbital Period': f'{PLANET.orbit_period.round(3):latex}{foot(REF["kagetani2023"])}',
    'Eccentricity': f'{PLANET.eccentricity:.1f}{foot(REF["assumed"])}',
    'Initial Phase': f'{PLANET.init_phase:latex}{foot(REF["assumed"])}',
    'Distance': f'{SYSTEM.distance:latex}{foot(REF["gaiacollaboration2020"])}',
    'Inclination': f'{SYSTEM.inclination:latex}{foot(REF["kagetani2023"])}',
    'Aperature': f'{INST.telescope.aperture.round(1):latex}',
    'Observation Length': f'{OBS.observation_time.round(1):latex}',
    'Observation Cadence': f'{OBS.integration_time:latex}',
    'Short Wavelength': f'{INST.bandpass.wl_blue:latex}',
    'Long Wavelength': f'{INST.bandpass.wl_red:latex}',
    'Resolving Power': f'{INST.bandpass.resolving_power:.0f}',
    'Mean Molecular Weight': f'{GCM_DICT["gcm"]["mean_molec_weight"]:.0f}$\\;\\mathrm{{g\\;cm^{{-3}}}}${foot(REF["assumed"])}',
    'Albedo': f'{GCM_DICT["gcm"]["vspec"]["albedo"]:.1f}{foot(REF["assumed"])}',
    'SW': f'$<\\;${SW_MAX.round(1):latex}',
    'LW': f'$>\\;${LW_MIN.round(1):latex}'
}

if __name__ == '__main__':
    model = get_model()
    if not (model.directories['psg_thermal'] / 'phase00000.fits').exists() or RERUN_PLANET:
        psg.docker.set_url_and_run()
        model.build_planet()
    if not (model.directories['all_model'] / 'phase00000.fits').exists() or RERUN_SPECTRA:
        model.build_spectra()
