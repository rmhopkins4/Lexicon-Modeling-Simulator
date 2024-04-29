import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import Levenshtein as lev

import random


def draw_graph(graph, color='blue', size=100):
    nx.draw(graph, with_labels=True, node_color=color, node_size=size)
    plt.show()


# Function to randomly generate initial strings for agents
def build_population_string(num_agents, string_lengths):
    return [''.join(
        random.choices('abcdefghijklmnopqrstuvwxyz',
                       k=random.choice(string_lengths)))
            for _ in range(num_agents)]


def create_regular_graph_w_pop_string(nodes, neighbors, string_lengths):
    G = nx.connected_watts_strogatz_graph(nodes, neighbors, 0)

    pop = build_population_string(
        num_agents=nodes, string_lengths=string_lengths)

    return G, pop


def create_2bell_bars_graph_w_pop_string(size_1, size_2, len_bar, num_bars, string_lengths):
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

    pop = build_population_string(num_agents=size_1 +
                                  size_2 + (len_bar * num_bars), string_lengths=string_lengths)

    return barbell_graph, pop


def create_complete_graph_w_pop_string(group1, string_lengths):
    G = nx.complete_graph(group1)

    pop = build_population_string(num_agents=group1,
                                  string_lengths=string_lengths)

    return G, pop


def create_bipartite_graph_w_pop_string(group_1, group_2, string_lengths):
    G = nx.bipartite.complete_bipartite_graph(group_1, group_2)

    pop = build_population_string(num_agents=group_1 + group_2,
                                  string_lengths=string_lengths)

    return G, pop


def create_lollipop_graph_w_pop_string(candy_nodes, stick_nodes, string_lengths):
    G = nx.lollipop_graph(candy_nodes, stick_nodes)

    pop = build_population_string(num_agents=candy_nodes + stick_nodes,
                                  string_lengths=string_lengths)

    return G, pop


def create_small_world_graph_w_pop_string(nodes, neighbors, rewire_odds, string_lengths):
    G = nx.connected_watts_strogatz_graph(nodes, neighbors, rewire_odds)

    pop = build_population_string(num_agents=nodes,
                                  string_lengths=string_lengths)

    return G, pop


def select_neighbor(graph, node: int):
    neighbors = list(graph.neighbors(node))
    return random.choice(neighbors)


# Function to update listener's string to reduce Levenshtein distance by one
def interact_string(speak_string, listen_string, ignore_odds=0.0, mess_up_odds=0.0):
    # Calculate the Levenshtein distance between the input and destination strings
    distance_before = lev.distance(listen_string, speak_string)
    if distance_before == 0 or ignore_odds > random.random():
        return listen_string

    # Get the edit operations needed to change input_string to destination_string
    edit_operations = lev.editops(listen_string, speak_string)

    updated_string = listen_string
    # Loop until a valid edit is made
    while True:
        # Randomly select an edit operation from the list
        selected_edit = random.choice(edit_operations)

        # perhaps change it
        if mess_up_odds > random.random():
            if selected_edit[0] != 'delete':
                # choose which to modify
                index_to_change = random.choice([1, 2])
                affected_string_len = len(
                    [listen_string, speak_string][index_to_change-1])
                new_val = selected_edit[index_to_change] + \
                    random.choice([-1, 1])

                new_val = max(0, min(new_val, (affected_string_len - 1)))

                # build new selected_edit
                if index_to_change == 1:
                    selected_edit = (
                        selected_edit[0], new_val, selected_edit[2])
                else:
                    selected_edit = (
                        selected_edit[0], selected_edit[1], new_val)

        updated_string = lev.apply_edit(
            [selected_edit], updated_string, speak_string)

        # Calculate the new Levenshtein distance after the edit
        # distance_after = lev.distance(updated_string, destination_string)

        # Ensure that the edit reduces the Levenshtein distance by 1
        # if distance_after == distance_before - 1:
        break

    return updated_string


def sim_step_string(interaction_graph, population, ignore_odds, mess_up_odds):
    indices = list(range(len(population)))
    random.shuffle(indices)
    for speaker in indices:
        # pick a random neighbor and interact
        selected_neighbor = select_neighbor(interaction_graph, speaker)
        population[selected_neighbor] = interact_string(population[speaker],
                                                        population[selected_neighbor],
                                                        ignore_odds, mess_up_odds)


# Main simulation function
def run_simulation_string(**kwargs):
    """

    Kwrgs:
        type (str):
        num_runs (int):
        string_lengths (list[int]):
        ignore_odds (float):
        mess_up_odds (float):

        nodes_a (int): first node count
        nodes_b (int): second node count, if needed
        neighbor (int): number of neighbors in regular/small_world. must be even
        rewire (float): odds to rewire edge in small_world graph
        len_bar (int): length of bars in barbells+ graph
        num_bars (int): number of bars in barbells+ graph

        show_graph (bool)

    """

    iterations_list = []
    for i in range(kwargs['num_runs']):
        run_iterations = 0

        match kwargs['type']:
            case 'regular':
                run_g, run_pop = create_regular_graph_w_pop_string(kwargs['nodes_a'], kwargs['neighbor'],
                                                                   kwargs['string_lengths'])
            case 'complete':
                run_g, run_pop = create_complete_graph_w_pop_string(kwargs['nodes_a'],
                                                                    kwargs['string_lengths'])
            case 'small_world':
                run_g, run_pop = create_small_world_graph_w_pop_string(kwargs['nodes_a'], kwargs['neighbor'], kwargs['rewire'],
                                                                       kwargs['string_lengths'])
            case 'barbells':
                run_g, run_pop = create_2bell_bars_graph_w_pop_string(kwargs['nodes_a'], kwargs['nodes_b'], kwargs['len_bar'], kwargs['num_bars'],
                                                                      kwargs['string_lengths'])
            case 'bipartite':
                run_g, run_pop = create_bipartite_graph_w_pop_string(kwargs['nodes_a'], kwargs['nodes_b'],
                                                                     kwargs['string_lengths'])
            case 'lollipop':
                run_g, run_pop = create_lollipop_graph_w_pop_string(kwargs['nodes_a'], kwargs['nodes_b'],
                                                                    kwargs['string_lengths'])

        if kwargs['show_graph']:
            # draw graph once
            draw_graph(run_g)
            kwargs['show_graph'] = False

        # average distance
        while not np.mean([lev.distance(run_pop[i], run_pop[j])
                           for i in range((len(run_pop)))
                           for j in range((len(run_pop)))]) == 0:

            # run a step
            sim_step_string(
                run_g, run_pop, kwargs['ignore_odds'], kwargs['mess_up_odds'])
            run_iterations += 1

        iterations_list.append(run_iterations)
        print(f"{len(iterations_list)}/{kwargs['num_runs']}")

    return iterations_list

    """
    agents = build_population_string(num_agents, string_length)
    neighbors = {i: [j for j in range(num_agents) if j != i]
                 for i in range(num_agents)}

    for iteration in range(max_iterations):
        # Randomly select an agent to update
        agent_index = random.randint(0, num_agents - 1)
        updated_string = interact_string(
            agents[agent_index], random.choice(agents), ignore_odds, mess_up_odds)  # !!

        # Update agent's string
        agents[agent_index] = updated_string

        # Check for convergence
        distances = [lev.distance(agents[i], agents[j]) for i in range(
            num_agents) for j in neighbors[i]]
        avg_distance = np.mean(distances)
        if avg_distance == 0:
            print("Convergence reached at iteration:", iteration)
            break

    return agents
    """


"""
# Example usage
num_agents = 10
string_length = [5, 6, 7]
max_iterations = 10000

final_strings = run_simulation_string(
    num_agents, string_length, max_iterations, 0.1, 0.1)
print("Final strings:", final_strings)
"""
