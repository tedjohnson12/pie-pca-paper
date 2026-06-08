"""
Color definitions to make pretty plots.
"""
from matplotlib.colors import LinearSegmentedColormap

TEAL = '#54B295'
LIGHT_ORANGE = '#DC972B'
DARK_ORANGE = '#E28221'
YELLOW = '#C6A82E'
BROWN = '#825F28'
SLATE = '#544F43'
BEIGE = '#807556'
CREAM = '#DCCA94'


cm_teal_cream_orange = LinearSegmentedColormap.from_list(
    'cm_teal_cream_orange', colors=[TEAL, CREAM, DARK_ORANGE])
