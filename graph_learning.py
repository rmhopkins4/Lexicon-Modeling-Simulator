import networkx as nx
import matplotlib.pyplot as plt

import random
import pprint

# RANDOM_SEED = 118755212
# random.seed(RANDOM_SEED)
NUM_SYMBOLS = 10


def create_regular_graph_w_pop(nodes, neighbors, num_symbols=NUM_SYMBOLS):
    G = nx.connected_watts_strogatz_graph(nodes, neighbors, 0)

    return G


def create_2bell_bars_graph_w_pop(size_1, size_2, len_bar, num_bars, num_symbols=NUM_SYMBOLS):
    # Create first completes
    bell1 = nx.complete_graph(range(size_1))
    bell2 = nx.complete_graph(range(size_1, size_1 + size_2))

    # Combine the two bells using disjoint_union
    barbell_graph = nx.disjoint_union(bell1, bell2)

    total_nodes = size_1 + size_2

    # add bars
    for i in range(num_bars):
        prev_node = random.choice(range(size_1))
        for j in range(len_bar):
            barbell_graph.add_node(total_nodes + (len_bar * i) + j)
            barbell_graph.add_edge(total_nodes + (len_bar * i) + j, prev_node)
            prev_node = (total_nodes + (len_bar * i) + j)

        barbell_graph.add_edge(prev_node,
                               random.choice(range(size_1, size_1 + size_2)))

    return barbell_graph


def create_bipartite_graph_w_pop(group_1, group_2, num_symbols=NUM_SYMBOLS):
    G = nx.bipartite.complete_bipartite_graph(group_1, group_2)

    return G


def create_lollipop_graph_w_pop(candy_nodes, stick_nodes, num_symbols=NUM_SYMBOLS):
    G = nx.lollipop_graph(candy_nodes, stick_nodes)

    return G


def create_small_world_graph_w_pop(nodes, neighbors, rewire_odds, num_symbols=NUM_SYMBOLS):
    G = nx.connected_watts_strogatz_graph(nodes, neighbors, rewire_odds)

    return G

# random drift (together) after interactions? more impact on barbell-type graphs, accents
#   depends on how i model current state, difficult to do w/ homesign method, which in all likelihood i will keep

# complete, random initial opinions, how long before consensus for different sizes?
# barbell, only two distinct initial agreements, tweak sizes (relative, total) to predict which wins and how long it takes?
# bipartite, compare w/ barbell
# lollipop, how long to fully agree depending on length of stick and size of candy?
# small world vs regular, how much does a small set of extra connections help w/ consensus time?
#

# barabási–albert?


g = create_small_world_graph_w_pop(10, 4, 0.1)
a = nx.to_numpy_array(g)
pprint.pp(a)
nx.draw(g, with_labels=True, node_color='pink', node_size=100)
plt.show()
