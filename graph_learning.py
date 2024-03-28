import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import random
import pprint

# RANDOM_SEED = 118755212
# random.seed(RANDOM_SEED)
NUM_SYMBOLS = 10
LEARN_COEFFICIENT = 0.2


def _build_population(population_count, num_symbols):
    population = np.random.rand(population_count, num_symbols)
    return np.divide(population, np.sum(population, axis=1, keepdims=True))


def create_regular_graph_w_pop(nodes, neighbors, num_symbols=NUM_SYMBOLS):
    G = nx.connected_watts_strogatz_graph(nodes, neighbors, 0)

    pop = _build_population(population_count=nodes, num_symbols=num_symbols)

    return G, pop


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

    pop = _build_population(population_count=size_1 + size_2 + (len_bar * num_bars), num_symbols=num_symbols)

    return barbell_graph, pop


def create_bipartite_graph_w_pop(group_1, group_2, num_symbols=NUM_SYMBOLS):
    G = nx.bipartite.complete_bipartite_graph(group_1, group_2)

    pop = _build_population(population_count=group_1 + group_2, num_symbols=num_symbols)

    return G, pop


def create_lollipop_graph_w_pop(candy_nodes, stick_nodes, num_symbols=NUM_SYMBOLS):
    G = nx.lollipop_graph(candy_nodes, stick_nodes)

    pop = _build_population(population_count=candy_nodes + stick_nodes, num_symbols=num_symbols)

    return G, pop


def create_small_world_graph_w_pop(nodes, neighbors, rewire_odds, num_symbols=NUM_SYMBOLS):
    G = nx.connected_watts_strogatz_graph(nodes, neighbors, rewire_odds)

    pop = _build_population(population_count=nodes, num_symbols=num_symbols)

    return G, pop


def distinctness(speaker, population):
    return np.sum(np.mean(np.abs(population - speaker), axis=1))


def select_neighbor(graph, node: int):
    neighbors = list(graph.neighbors(node))   
    return random.choice(neighbors)


def interact(speaker, listener):
    performed_symbol = np.random.choice(len(speaker), p=speaker)
    # non-performed symbols become less common
    listener[np.arange(len(listener) != performed_symbol)
             ] *= (1-LEARN_COEFFICIENT)
    # performed symbol becomes more common
    listener[performed_symbol] += LEARN_COEFFICIENT * (1-LEARN_COEFFICIENT)
    # make sure sum is still 1
    listener /= np.sum(listener)


def run_simulation_step(interaction_graph, population):
    # all speakers interact with one random neighbor of theirs, in a random order
    indices = list(range(len(population)))
    random.shuffle(indices)
    for speaker in indices:
        # pick a random neighbor and interact w/ them
        interact(population[speaker], population[select_neighbor(interaction_graph, speaker)])


# random drift (together) after interactions? more impact on barbell-type graphs, accents
#   depends on how i model current state, difficult to do w/ homesign method, which in all likelihood i will keep

# complete, random initial opinions, how long before consensus for different sizes?
# barbell, only two distinct initial agreements, tweak sizes (relative, total) to predict which wins and how long it takes?
# bipartite, compare w/ barbell
# lollipop, how long to fully agree depending on length of stick and size of candy?
# small world vs regular, how much does a small set of extra connections help w/ consensus time?
#

# barabási–albert?


g, pop = create_regular_graph_w_pop(10, 4)
print(np.mean([distinctness(pop[i], pop) for i in range(len(pop))]))

for i in range(10000):
    run_simulation_step(g, pop)
    print(np.mean([distinctness(pop[i], pop) for i in range(len(pop))]))
    
print(pop)

# nx.draw(g, with_labels=True, node_color='pink', node_size=100)
# plt.show()
