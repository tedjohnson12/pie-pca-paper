"""
Get a dictionary to be used as a column in the table.
"""
from astropy import units as u
import mirecle50cm_run as m050
import mirecle2m_run as m200
import mirecle564cm_run as m564
import proxb_run as proxb
from proxb_run import REF
import mirecle_inference as mi
from common import foot

mods = [m050, m200, m564]


TAB = {
    'Stellar Effective Temperature': [f'{proxb.TEFF} K{foot(REF["faria2022"])}' for m in mods],
    'Stellar Radius': [f'{m.STAR.radius:latex}{foot(REF["faria2022"])}' for m in mods],
    'Stellar Rotation Period': [f'{m.STAR.period:latex}{foot(REF["faria2022"])}' for m in mods],
    'Spot Temperature': [f'{m.STAR.spots.teff_umbra:latex}{foot(REF["assumed"])}' for m in mods],
    'Spot Coverage Fraction': [f'{proxb.SPOT_FRAC:.1f}{foot(REF["assumed"])}' for m in mods],
    'Photosphere Temperature': [f'{m.STAR.teff.round(0):latex}' for m in mods],
    'Planet Radius': [f'{m.PLANET.radius.round(2):latex}{foot(REF["assumed"])}' for m in mods],
    'Planet Mass': [f'{m.PLANET.gravity.value.to(u.M_earth).round(0):latex}{foot(REF["assumed"])}' for m in mods],
    'Planet $T_\\mathrm{eq}$': [f'{m.get_teq().to(u.K).round(0):latex}' for m in mods],
    'Semimajor Axis': [f'{m.PLANET.semimajor_axis:latex}{foot(REF["faria2022"])}' for m in mods],
    'Orbital Period': [f'{m.PLANET.orbit_period.round(3):latex}{foot(REF["faria2022"])}' for m in mods],
    'Eccentricity': [f'{m.PLANET.eccentricity:.1f}{foot(REF["faria2022"])}' for m in mods],
    'Initial Phase': [f'{m.PLANET.init_phase:latex}{foot(REF["assumed"])}' for m in mods],
    'Distance': [f'{m.SYSTEM.distance:latex}{foot(REF["gaiacollaboration2020"])}' for m in mods],
    'Inclination': [f'{m.SYSTEM.inclination:latex}{foot(REF["assumed"])}' for m in mods],
    'Aperature': [f'{m.INST.telescope.aperture.round(1):latex}' for m in mods],
    'Observation Length': [f'{m.OBS.observation_time.round(1):latex}' for m in mods],
    'Observation Cadence': [f'{m.OBS.integration_time:latex}' for m in mods],
    'Short Wavelength': [f'{m.INST.bandpass.wl_blue:latex}' for m in mods],
    'Long Wavelength': [f'{m.INST.bandpass.wl_red:latex}' for m in mods],
    'Resolving Power': [f'{m.INST.bandpass.resolving_power:.0f}' for m in mods],
    'Mean Molecular Weight': [f'{m.GCM_DICT["gcm"]["mean_molec_weight"]:.0f}$\\;\\mathrm{{g\\;cm^{{-3}}}}${foot(REF["assumed"])}' for m in mods],
    'Albedo': [f'{m.GCM_DICT["gcm"]["vspec"]["albedo"]:.1f}{foot(REF["assumed"])}' for m in mods],
    'SW': [f'$<\\;${mi.SW_MAX.round(1):latex}' for m in mods],
    'LW': [f'$>\\;${mi.LW_MIN.round(1):latex}' for m in mods],
}

def get_refined_table():
    new_tab = {}
    for k, v in TAB.items():
        matches_proxb = [v[i] == proxb.TAB[k] for i in range(len(v))]
        if all(matches_proxb):
            new_tab[k] = '-'
        else:
            all_same = [v[i] == v[0] for i in range(len(v))]
            if all(all_same):
                new_tab[k] = v[0]
            else:
                new_tab[k] = ', '.join(
                    ['-' if v[i] == proxb.TAB[k] else v[i] for i in range(len(v))]
                )
    return new_tab


if __name__ == '__main__':
    print(get_refined_table())