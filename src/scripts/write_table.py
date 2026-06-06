"""
Combine everything into one table.
"""

from loguru import logger
from astropy import units as u

import toi519_run as toi519
import gj876_run as gj876
import proxb_run as proxb
import paths

TABLE_FILE = paths.output / 'tab.txt'

def write_table():
    lines = [
        '\\begin{table*}',
        '\\centering',
        '\\begin{tabular}{cc}',
        '\\hline',
        'Quantity & TOI-519 b & GJ 876 d & PCb \\\\',
        '\\hline',
    ]
    keys_toi519 = toi519.TAB.keys()
    keys_gj876 = gj876.TAB.keys()
    keys_proxb = proxb.TAB.keys()
    for k in keys_toi519 & keys_gj876 & keys_proxb:
        try:
            assert k in toi519.TAB
            assert k in gj876.TAB
            assert k in proxb.TAB
        except AssertionError:
            logger.warning(f'{k} not in all tables')
        lines.append(f'{k} & {toi519.TAB.get(k,"")} & {gj876.TAB.get(k,"")} & {proxb.TAB.get(k,"")} \\\\')
    lines.append('\\hline')
    lines.append('\\multicolumn{4}{c}{Inference Grid} \\\\')
    lines.append('\\hline')
    _radius_range = [
        f'${toi519.RADIUS_SCALE_MIN*toi519.PLANET.radius.to_value(u.R_jup):.1f}--{toi519.RADIUS_SCALE_MAX*toi519.PLANET.radius.to_value(u.R_jup):.1f}\\,R_\\mathrm{{J}}$'
        f'${gj876.RADIUS_SCALE_MIN*gj876.PLANET.radius.to_value(u.R_earth):.1f}--{gj876.RADIUS_SCALE_MAX*gj876.PLANET.radius.to_value(u.R_earth):.1f}\\,R_\\oplus$'
        f'${proxb.RADIUS_SCALE_MIN*proxb.PLANET.radius.to_value(u.R_earth):.1f}--{proxb.RADIUS_SCALE_MAX*proxb.PLANET.radius.to_value(u.R_earth):.1f}\\,R_\\oplus$'
    ]
    _temp_range = [
        f'${toi519.TEMP_RATIO_MIN:.2f}--{toi519.TEMP_RATIO_MAX:.2f}$'
        f'${gj876.TEMP_RATIO_MIN:.2f}--{gj876.TEMP_RATIO_MAX:.2f}$'
        f'${proxb.TEMP_RATIO_MIN:.2f}--{proxb.TEMP_RATIO_MAX:.2f}$'
    ]
    lines.append(f'Radius & {" & ".join(_radius_range)} \\\\')
    lines.append(f'$T_\\mathrm{{night}}/T_\\mathrm{{day}}$ & {" & ".join(_temp_range)} \\\\')
    lines.append('\\hline')

    lines.append('\\end{tabular}')
    lines.append('\\caption{Simulation Parameters}')
    lines.append('\\label{tab:parameters}')
    lines.append('\\end{table*}')

    with open(TABLE_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

if __name__ == '__main__':
    write_table()