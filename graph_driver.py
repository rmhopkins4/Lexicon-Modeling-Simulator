import argparse
from argparse import RawTextHelpFormatter
import textwrap

import graph_learning as gl

NUM_SYMBOLS = 10
LEARN_COEFFICIENT = 0.2
DISTINCTNESS_THRESHOLD = 1e-10

parser = argparse.ArgumentParser(
    prog='',
    description='Lexicon consensus modeling using graphs',
    epilog='',
    formatter_class=RawTextHelpFormatter  # fixes newline issues
)

# type, number of runs, learn_coefficient kwargs(nodes_a, nodes_b, neighbors, num_symbols, len_bar, num_bars, ...)
parser.add_argument('type', action='store', type=str, help=textwrap.dedent(
    """    each 'type' requires different keyword arguments, listed below
    regular: 'nodes_a', 'neighbors'
    small_world: 'nodes_a', 'neighbors', 'rewire_odds'
    barbells: 'nodes_a', 'nodes_b', 'len_bar', 'num_bars'
    bipartite: 'nodes_a', 'nodes_b'
    lollipop: 'nodes_a', 'nodes_b'"""))
parser.add_argument('num_runs', action='store', type=int,
                    help='number of distinct simulations, averaged')
parser.add_argument('num_symbols', nargs='?', action='store', help='number of possible symbols',
                    type=int, default=NUM_SYMBOLS)
parser.add_argument('l_coefficient', nargs='?', action='store', help='how quickly an agent learns',
                    type=float, default=LEARN_COEFFICIENT)
parser.add_argument('distinct_thresh', nargs='?', action='store', help='threshold for considering agents\' idiolects distinct',
                    type=float, default=DISTINCTNESS_THRESHOLD)
parser.add_argument('--nodes_a', action='store', type=int,
                    help='number of nodes in primary structure')
parser.add_argument('--nodes_b', action='store', type=int,
                    help='number of nodes in secondary structure (if applicable)')
parser.add_argument('--neighbor', action='store', type=int,
                    help='custom number of neighbors (if applicable), must be even')
parser.add_argument('--rewire', action='store', type=float,
                    help='odds to rewire edge small_world graph')
parser.add_argument('--bar_len', action='store', type=int,
                    help='length of bar in barbells+ graph')
parser.add_argument('--num_bars', action='store', type=int,
                    help='number of bars in barbells+ graph')
parser.add_argument('-show_graph', action='store_true',
                    help='toggles showing (one example) graph before simulations')

args = vars(parser.parse_args())

print(args)
