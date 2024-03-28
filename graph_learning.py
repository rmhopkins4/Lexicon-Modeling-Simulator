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

    pop = _build_population(population_count=size_1 +
                            size_2 + (len_bar * num_bars), num_symbols=num_symbols)

    return barbell_graph, pop


def create_bipartite_graph_w_pop(group_1, group_2, num_symbols=NUM_SYMBOLS):
    G = nx.bipartite.complete_bipartite_graph(group_1, group_2)

    pop = _build_population(population_count=group_1 +
                            group_2, num_symbols=num_symbols)

    return G, pop


def create_lollipop_graph_w_pop(candy_nodes, stick_nodes, num_symbols=NUM_SYMBOLS):
    G = nx.lollipop_graph(candy_nodes, stick_nodes)

    pop = _build_population(population_count=candy_nodes +
                            stick_nodes, num_symbols=num_symbols)

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
        interact(population[speaker], population[select_neighbor(
            interaction_graph, speaker)])


# random drift (together) after interactions? more impact on barbell-type graphs, accents
#   depends on how i model current state, difficult to do w/ homesign method, which in all likelihood i will keep

# complete, random initial opinions, how long before consensus for different sizes?
# barbell, only two distinct initial agreements, tweak sizes (relative, total) to predict which wins and how long it takes?
# bipartite, compare w/ barbell
# lollipop, how long to fully agree depending on length of stick and size of candy?
# small world vs regular, how much does a small set of extra connections help w/ consensus time?
#

# barabási–albert?


def run_simulation(type: str, num_runs: int, **kwargs):
    """_summary_

    Args:
        type (str): which graph are you making?
        num_runs (int): how many times do we run it?
    Kwargs:
        nodes_a (int): first node count
        nodes_b (int): second node count, if needed
        neighbors (int): number of neighbors in regular/small_world. must be even  
        len_bar (int): length of bars in barbells+
        num_bars (int): number of bars in barbells+
        num_symbols (int): optional, defaults to NUM_SYMBOLS constant 
    """

    distinctnesses = []  # 2d array
    for i in range(num_runs):
        run_distinctnesses = []

        match type:
            case 'regular':
                run_g, run_pop = create_regular_graph_w_pop(kwargs['nodes_a'], kwargs['neighbors'],
                                                            kwargs['num_symbols'] if 'num_symbols' in kwargs else NUM_SYMBOLS)
            case 'small_world':
                run_g, run_pop = create_small_world_graph_w_pop(kwargs['nodes_a'], kwargs['neighbors'], kwargs['rewire_odds'],
                                                                kwargs['num_symbols'] if 'num_symbols' in kwargs else NUM_SYMBOLS)
            case 'barbells':
                run_g, run_pop = create_2bell_bars_graph_w_pop(kwargs['nodes_a'], kwargs['nodes_b'], kwargs['len_bar'], kwargs['num_bars'],
                                                               kwargs['num_symbols'] if 'num_symbols' in kwargs else NUM_SYMBOLS)
            case 'bipartite':
                run_g, run_pop = create_bipartite_graph_w_pop(kwargs['nodes_a'], kwargs['nodes_b'],
                                                              kwargs['num_symbols'] if 'num_symbols' in kwargs else NUM_SYMBOLS)
            case 'lollipop':
                run_g, run_pop = create_lollipop_graph_w_pop(kwargs['nodes_a'], kwargs['nodes_b'],
                                                             kwargs['num_symbols'] if 'num_symbols' in kwargs else NUM_SYMBOLS)

        while not np.allclose(run_pop, run_pop[0], atol=1e-10):
            run_simulation_step(run_g, run_pop)

            run_distinctnesses.append(
                (np.mean([distinctness(run_pop[k], run_pop) for k in range(len(run_pop))])))

        distinctnesses.append(run_distinctnesses)
        print(f"{len(distinctnesses)}/{num_runs}")

    return distinctnesses


d = run_simulation(type='bipartite', num_runs=50,
                   nodes_a=1, nodes_b=19, neighbors=4, rewire_odds=0.2)

print([len(a) for a in d])
print(np.mean([len(a) for a in d]))


# nx.draw(g, with_labels=True, node_color='pink', node_size=100)
# plt.show()
