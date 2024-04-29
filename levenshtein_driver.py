import argparse
from argparse import RawTextHelpFormatter
import textwrap

import levenshtein_learning as ll

import numpy as np

import ast

STRING_LENGTHS = [5, 6, 7]
IGNORE_ODDS = 0.1
MESS_UP_ODDS = 0.1

parser = argparse.ArgumentParser(
    prog='',
    description='Lexicon consensus modeling on strings using graphs',
    epilog='',
    formatter_class=RawTextHelpFormatter  # fixes newline issues
)


def parse_list(input_str):
    try:
        return ast.literal_eval(input_str)
    except (SyntaxError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid list format: {input_str}")


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
parser.add_argument('string_lengths', action='store', help='length of strings [HAS DEFAULT]',
                    type=parse_list, default=STRING_LENGTHS)
parser.add_argument('ignore_odds', nargs='?', action='store', help='odds an interaction is ignored [HAS DEFAULT]',
                    type=float, default=IGNORE_ODDS)
parser.add_argument('mess_up_odds', nargs='?', action='store', help='odds an interaction is messed-up [HAS DEFAULT]',
                    type=float, default=MESS_UP_ODDS)
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
# print(args)
output = ll.run_simulation_string(**args)

print(output)  # len is number of steps taken
# mean number of steps taken for all sims
print(np.mean(output))
