"""
Color definitions to make pretty plots.
"""
from matplotlib.colors import LinearSegmentedColormap

teal = '#54B295'
light_orange = '#DC972B'
dark_orange = '#E28221'
yellow = '#C6A82E'
brown = '#825F28'
slate = '#544F43'
beige = '#807556'
cream = '#DCCA94'


cm_teal_cream_orange = LinearSegmentedColormap.from_list('cm_teal_cream_orange',colors=[teal,cream,dark_orange])

state_colors = {
    'u' : brown,
    'p': yellow,
    'r': dark_orange,
    'l': teal
}