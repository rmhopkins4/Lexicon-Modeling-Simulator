import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import random
import pprint


def draw_graph(graph, color='green', size=100, abort=False):
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos=pos, with_labels=True, node_color=color, node_size=size)
    if not abort:
        plt.show()


def _build_population(population_count, num_symbols):
    population = np.random.rand(population_count, num_symbols)
    return np.divide(population, np.sum(population, axis=1, keepdims=True))


def create_regular_graph_w_pop(nodes, neighbors, num_symbols):
    G = nx.connected_watts_strogatz_graph(nodes, neighbors, 0)

    pop = _build_population(population_count=nodes, num_symbols=num_symbols)

    return G, pop


def create_2bell_bars_graph_w_pop(size_1, size_2, len_bar, num_bars, num_symbols):
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


def create_complete_graph_w_pop(group1, num_symbols):
    G = nx.complete_graph(group1)

    pop = _build_population(population_count=group1,
                            num_symbols=num_symbols)

    return G, pop


def create_bipartite_graph_w_pop(group_1, group_2, num_symbols):
    G = nx.bipartite.complete_bipartite_graph(group_1, group_2)

    pop = _build_population(population_count=group_1 +
                            group_2, num_symbols=num_symbols)

    return G, pop


def create_lollipop_graph_w_pop(candy_nodes, stick_nodes, num_symbols):
    G = nx.lollipop_graph(candy_nodes, stick_nodes)

    pop = _build_population(population_count=candy_nodes +
                            stick_nodes, num_symbols=num_symbols)

    return G, pop


def create_small_world_graph_w_pop(nodes, neighbors, rewire_odds, num_symbols):
    G = nx.connected_watts_strogatz_graph(nodes, neighbors, rewire_odds)

    pop = _build_population(population_count=nodes, num_symbols=num_symbols)

    return G, pop


def distinctness(speaker, population):
    return np.sum(np.mean(np.abs(population - speaker), axis=1))


def select_neighbor(graph, node: int):
    neighbors = list(graph.neighbors(node))
    return random.choice(neighbors)


def interact(speaker, listener, learn_coefficient):
    performed_symbol = np.random.choice(len(speaker), p=speaker)

    # Linear-Reward-Penalty

    # non-performed symbols become less common
    listener[np.arange(len(listener) != performed_symbol)
             ] *= (1-learn_coefficient)
    # performed symbol becomes more common
    listener[performed_symbol] += (learn_coefficient)  # *
    #                              (1 - listener[performed_symbol]))
    # make sure sum is still 1
    listener /= np.sum(listener)


def sim_step(interaction_graph, population, learn_coefficient):
    # all speakers interact with one random neighbor of theirs, in a random order
    indices = list(range(len(population)))
    random.shuffle(indices)
    for speaker in indices:
        # pick a random neighbor and interact w/ them
        interact(population[speaker],
                 population[select_neighbor(
                     interaction_graph, speaker)],
                 learn_coefficient)


def run_simulation(**kwargs):
    """run simulation suite according to keyword arguments from user
    returns distinctnesses for each step of each sim 

    Kwargs:
        type (str): type of graph
        num_runs (int): number of distinct simulations, averaged?
        num_symbols (int): number of possible symbols
        l_coefficient (float): how quickly an agent learns
        distinct_thresh (float): threshold for considering agents\' idiolects distinct

        nodes_a (int): first node count
        nodes_b (int): second node count, if needed
        neighbor (int): number of neighbors in regular/small_world. must be even
        rewire (float): odds to rewire edge in small_world graph
        len_bar (int): length of bars in barbells+ graph
        num_bars (int): number of bars in barbells+ graph

        show_graph (bool): generate a bonus graph?
        debug (bool):
        metrics (bool):
    """

    iterations_list = []  # 2d array
    clusterings_list = []  # array
    s_paths_list = []  # array
    for i in range(kwargs['num_runs']):
        run_distinctnesses = []

        match kwargs['type']:
            case 'regular':
                run_g, run_pop = create_regular_graph_w_pop(kwargs['nodes_a'], kwargs['neighbor'],
                                                            kwargs['num_symbols'])
            case 'complete':
                run_g, run_pop = create_complete_graph_w_pop(kwargs['nodes_a'],
                                                             kwargs['num_symbols'])
            case 'small_world':
                run_g, run_pop = create_small_world_graph_w_pop(kwargs['nodes_a'], kwargs['neighbor'], kwargs['rewire'],
                                                                kwargs['num_symbols'])
            case 'barbells':
                run_g, run_pop = create_2bell_bars_graph_w_pop(kwargs['nodes_a'], kwargs['nodes_b'], kwargs['len_bar'], kwargs['num_bars'],
                                                               kwargs['num_symbols'])
            case 'bipartite':
                run_g, run_pop = create_bipartite_graph_w_pop(kwargs['nodes_a'], kwargs['nodes_b'],
                                                              kwargs['num_symbols'])
            case 'lollipop':
                run_g, run_pop = create_lollipop_graph_w_pop(kwargs['nodes_a'], kwargs['nodes_b'],
                                                             kwargs['num_symbols'])

        if kwargs['show_graph']:
            # draw graph once
            draw_graph(run_g)
            kwargs['show_graph'] = False

        clusterings_list.append(nx.average_clustering(run_g))
        s_paths_list.append(nx.average_shortest_path_length(run_g))

        while not np.all([np.allclose(row, run_pop[0], atol=kwargs['distinct_thresh']) for row in run_pop]):

            sim_step(run_g, run_pop, kwargs['l_coefficient'])

            if kwargs['debug']:
                print(run_pop)

            run_distinctnesses.append(
                (np.mean([distinctness(run_pop[k], run_pop) for k in range(len(run_pop))])))

        iterations_list.append(len(run_distinctnesses))
        if kwargs['debug']:
            print(f"{len(iterations_list)}/{kwargs['num_runs']}")

    if kwargs['metrics']:
        return iterations_list, clusterings_list, s_paths_list
    else:
        return iterations_list, None, None

# random drift (together) after interactions? more impact on barbell-type graphs, accents
#   depends on how i model current state, difficult to do w/ homesign method, which in all likelihood i will keep

# complete, random initial opinions, how long before consensus for different sizes?
# barbell, only two distinct initial agreements, tweak sizes (relative, total) to predict which wins and how long it takes?
# bipartite, compare w/ barbell
# lollipop, how long to fully agree depending on length of stick and size of candy?
# small world vs regular, how much does a small set of extra connections help/hurt consensus time?
#

# barabási–albert?
