"""
Simulate an observation of a hot sub-Neptune with JWST
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


TABLE_FILE = paths.output / 'proxb.txt'
TRUE_EPSILON = 0.1
TRUE_DAY_NIGHT_RATIOS = [0.1, 0.5, 0.9]

TEFF = 2900
SPOT_FRAC = 0.2
TSPOT = 2600
# (1-SPOT_FRAC)*TPHOT**4 + SPOT_FRAC*TSPOT**4 = TEFF**4
TPHOT = ((TEFF**4 - SPOT_FRAC*TSPOT**4)/(1-SPOT_FRAC))**0.25
SHORT_WL_CUTOFF = 0.8*u.um


HEADER = VSPEC.params.Header(
    data_path=Path(__file__).parent / '.vspec' / f'proxb_{TRUE_EPSILON:.2f}',
    seed=11,
    spec_grid=VSPEC.params.VSPECGridParameters(
        max_teff=3100 * u.K,
        min_teff=2600 * u.K,
        impl_bin='rust',
        impl_interp='scipy',
        fail_on_missing=False
    ),
    desc='Proxb with JWST',
)

STAR = VSPEC.params.StarParameters(
    psg_star_template='M',
    teff=TPHOT * u.K,
    radius=0.141 * u.R_sun,
    period=90 * u.day,
    misalignment=0 * u.deg,
    misalignment_dir=0 * u.deg,
    mass=0.15 * u.Msun,
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
    name='proxcenb',
    radius=1.0*u.R_earth,
    gravity=VSPEC.params.GravityParameters(
        mode='kg',
        value=1.0*u.M_earth
    ),
    semimajor_axis=0.04856*u.AU,
    orbit_period=11.1868*u.day,
    rotation_period=11.19*u.day,
    eccentricity=0.0,
    obliquity=0.0*u.deg,
    obliquity_direction=0.0*u.deg,
    init_substellar_lon=0.0*u.deg,
    init_phase=90*u.deg,
)

SYSTEM = VSPEC.params.SystemParameters(
    distance=1.302*u.pc,
    inclination=80*u.deg,
    phase_of_periastron=0*u.deg
)

OBS = VSPEC.params.ObservationParameters(
    observation_time=11.19*u.day,
    integration_time=4*u.hr
)

PSG = VSPEC.params.psgParameters(
    gcm_binning=200,
    phase_binning=1,
    use_continuum_stellar=True,
    use_molecular_signatures=True,
    nmax=2,
    lmax=1,
    continuum=['Rayleigh', 'CIA_all', 'Refraction']
)

INST = VSPEC.params.InstrumentParameters.miri_lrs()
# INST = VSPEC.params.InstrumentParameters(
#     telescope=VSPEC.params.SingleDishParameters(
#         aperture=2*u.m,
#         zodi=1.0
#     ),
#     bandpass=VSPEC.params.BandpassParameters(
#         wl_blue=1*u.um,
#         wl_red=18*u.um,
#         resolving_power=50,
#         wavelength_unit=u.micron,
#         flux_unit=u.Unit('W m-2 um-1')
#     ),
#     detector=VSPEC.params.DetectorParameters(
#         beam_width=0.5*u.arcsec,
#         integration_time=10*u.s,
#         ccd=VSPEC.params.ccdParameters(
#             pixel_sampling=1,
#             read_noise=16.8*u.electron,
#             dark_current=100*u.electron/u.s,
#             throughput=0.7,
#             emissivity=0.1,
#             temperature=35*u.K
#         )
#     )
# )
GCM_DICT = {
    'star': {
        'teff': STAR.teff,
        'radius': STAR.radius
    },
    'planet': {
        'semimajor_axis': PLANET.semimajor_axis,
    },
    'gcm': {
        'mean_molec_weight': 28.0,
        'vspec': {
            'nlayer': 30,
            'nlon': 90,
            'nlat': 45,
            'epsilon': TRUE_EPSILON,
            'psurf': 1*u.bar,
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
    return VSPEC.ObservationModel(VSPEC_PARAMS)

def get_teq():
    return STAR.teff * np.sqrt(STAR.radius / PLANET.semimajor_axis/2) * (1-GCM_DICT['gcm']['vspec']['albedo'])
def foot(t):
    return rf'$^{t}$'

REF = {
    'assumed': '\\dagger',
    'faria2022': 'a',
    'gaiacollaboration2020': 'b'
    
}
def cite(k):
    if k in ['assumed']:
        return k
    else:
        return f'\\citet{{{k}}}'


def write_table():
    tab = {
        'Stellar Effective Temperature': f'{STAR.teff:latex}{foot(REF["faria2022"])}',
        'Stellar Radius': f'{STAR.radius:latex}{foot(REF["faria2022"])}',
        'Stellar Rotation Period': f'{STAR.period:latex}{foot(REF["faria2022"])}',
        'Spot Temperature': f'{STAR.spots.teff_umbra:latex}{foot(REF["assumed"])}',
        'Spot Coverage': f'{SPOT_FRAC:.1f}{foot(REF["assumed"])}',
        'Photosphere Temperature': f'{STAR.teff.round(0):latex}',
        'Planet Radius': f'{PLANET.radius.round(2):latex}{foot(REF["assumed"])}',
        'Planet Mass': f'{PLANET.gravity.value.to(u.M_earth).round(0):latex}{foot(REF["assumed"])}',
        'Planet $T_\\mathrm{eq}$': f'{get_teq().to(u.K).round(0):latex}',
        'Semimajor Axis': f'{PLANET.semimajor_axis:latex}{foot(REF["faria2022"])}',
        'Orbital Period': f'{PLANET.orbit_period.round(3):latex}{foot(REF["faria2022"])}',
        'Eccentricity': f'{PLANET.eccentricity:.1f}{foot(REF["faria2022"])}',
        'Initial Phase': f'{PLANET.init_phase:latex}{foot(REF["assumed"])}',
        'Distance': f'{SYSTEM.distance:latex}{foot(REF["gaiacollaboration2020"])}',
        'Inclination': f'{SYSTEM.inclination:latex}{foot(REF["assumed"])}',
        'Observation Length': f'{OBS.observation_time:latex}',
        'Integration Length': f'{INST.detector.integration_time:latex}',
        'Time Bin Size': f'{OBS.integration_time:latex}',
        'Short Wavelength': f'{INST.bandpass.wl_blue:latex}',
        'Long Wavelength': f'{INST.bandpass.wl_red:latex}',
        'PIE Cutoff': f'{SHORT_WL_CUTOFF:latex}',
        'Resolving Power': f'{INST.bandpass.resolving_power:.0f}',
        'Mean Molecular Weight': f'{GCM_DICT["gcm"]["mean_molec_weight"]:.0f}{foot(REF["assumed"])}',
        'Albedo': f'{GCM_DICT["gcm"]["vspec"]["albedo"]:.1f}{foot(REF["assumed"])}',
    }
    lines = [
        '\\begin{table}',
        '\\centering',
        '\\begin{tabular}{cc}',
        '\\hline',
        'Quantity & Value \\\\',
        '\\hline',
    ]
    for k, v in tab.items():
        lines.append(f'{k} & {v} \\\\')
    lines.append('\\hline')
    lines.append('\\end{tabular}')
    refsline = '; '.join(f"{foot(b)}{cite(a)}" for a,b in REF.items())
    lines.append(f'\\caption{{PCb Simulation Parameters. {refsline}}}')
    lines.append('\\label{tab:pcb-parameters}')
    lines.append('\\end{table}')

    with open(TABLE_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def get_temperature_ratio(epsilon: float):
    if epsilon < 1:
        mode = 'ivp_reflect'
    elif epsilon < 10:
        mode = 'bvp'
    else:
        mode = 'analytic'
    _, tsurf = VSPEC.gcm.heat_transfer.get_equator_curve(epsilon, 180, mode)
    return np.min(tsurf)/np.max(tsurf)


if __name__ == '__main__':
    write_table()
    psg.docker.set_url_and_run()
    model = get_model()
    model.build_planet()
    model.build_spectra()
