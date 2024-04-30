import argparse
from argparse import RawTextHelpFormatter
import textwrap

import graph_learning as gl

import numpy as np

NUM_SYMBOLS = 10
LEARN_COEFFICIENT = 0.01
DISTINCTNESS_THRESHOLD = 1e-8

parser = argparse.ArgumentParser(
    prog='',
    description='Lexicon consensus modeling using graphs',
    epilog='',
    formatter_class=RawTextHelpFormatter  # fixes newline issues
)

# type, number of runs, learn_coefficient kwargs(nodes_a, nodes_b, neighbors, num_symbols, len_bar, num_bars, ...)
parser.add_argument('type', action='store', type=str, help=textwrap.dedent(
    """    each 'type' requires different keyword arguments, listed below
    regular: 'nodes_a', 'neighbor'
    complete: 'nodes_a'
    small_world: 'nodes_a', 'neighbor', 'rewire'
    barbells: 'nodes_a', 'nodes_b', 'len_bar', 'num_bars'
    bipartite: 'nodes_a', 'nodes_b'
    lollipop: 'nodes_a', 'nodes_b'"""))
parser.add_argument('num_runs', action='store', type=int,
                    help='number of distinct simulations, averaged')
parser.add_argument('num_symbols', nargs='?', action='store', help='number of possible symbols [HAS DEFAULT]',
                    type=int, default=NUM_SYMBOLS)
parser.add_argument('l_coefficient', nargs='?', action='store', help='how quickly an agent learns [HAS DEFAULT]',
                    type=float, default=LEARN_COEFFICIENT)
parser.add_argument('distinct_thresh', nargs='?', action='store', help='threshold for considering agents\' idiolects distinct [HAS DEFAULT]',
                    type=float, default=DISTINCTNESS_THRESHOLD)
parser.add_argument('--nodes_a', action='store', type=int,
                    help='number of nodes in primary structure')
parser.add_argument('--nodes_b', action='store', type=int,
                    help='number of nodes in secondary structure (if applicable)')
parser.add_argument('--neighbor', action='store', type=int,
                    help='custom number of neighbors (if applicable), must be even')
parser.add_argument('--rewire', action='store', type=float,
                    help='odds to rewire edge small_world graph')
parser.add_argument('--len_bar', action='store', type=int,
                    help='length of bar in barbells+ graph')
parser.add_argument('--num_bars', action='store', type=int,
                    help='number of bars in barbells+ graph')
parser.add_argument('-show_graph', action='store_true',
                    help='toggles showing (one example) graph before simulations')
parser.add_argument('-debug', action='store_true',
                    help="toggle prints on")
parser.add_argument('-metrics', action='store_true',
                    help='toggle advanced metrics returned as tuple')

args = vars(parser.parse_args())
iterations, clusterings, shortest_paths = gl.run_simulation(**args)

# print([len(a) for a in distinctnesses])  # len is number of steps taken
# mean number of steps taken for all sims
print({
    'iterations': iterations,
    'clusterings': clusterings,
    'shortest_paths': shortest_paths
})
