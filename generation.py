import Levenshtein as lev
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os

import data
import levenshtein_learning as ll


env = os.environ.copy()
venv_path = ".venv"
env['PYTHONPATH'] = os.path.join(
    venv_path, 'Lib', 'site-packages') + os.pathsep + env.get('PYTHONPATH', '')


def call_command(command: list[str]):
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr)

    return result.stdout


"""
test_command_lev = ['python', 'levenshtein_driver.py',
                    'complete', '100', '[5, 6, 7]',
                    '0.1', '0.1', '--nodes_a', '10']

test_command_proprtion = ['python', 'graph_driver.py',
                          'complete', '100', '10',
                          '0.2', '--nodes_a', '10']
"""


def string_cross_mess_up_3d():
    # complete graph
    ignore_cross_messup_3d = []
    for ignore in range(10):
        ignore_row = []
        for mess_up in range(10):
            ignore_row.append(float(call_command(
                ['python', 'levenshtein_driver.py',
                 'complete', '100', '[5, 6, 7]',
                 str(0.1 * ignore), str(0.1 * mess_up), '--nodes_a', '10']
            )))
            print(ignore, mess_up)
        ignore_cross_messup_3d.append(ignore_row)

    # data.plot_3d_mesh(ignore_cross_messup_3d)
    data.plot_3d_bars(ignore_cross_messup_3d,
                      "ignore_odds (x0.1)", "mess_up_odds (x0.1)", "# simulations",
                      "string consensus, 100 runs, [5, 6, 7] length\n10-node complete graph")
    plt.savefig('figures/string_mistakes_complete_3D.png')

    # all-to-one bipartite
    ignore_cross_messup_3d = []
    for ignore in range(10):
        ignore_row = []
        for mess_up in range(10):
            ignore_row.append(float(call_command(
                ['python', 'levenshtein_driver.py',
                 'bipartite', '30', '[5, 6, 7]',
                 str(0.1 * ignore), str(0.1 * mess_up), '--nodes_a', '9', '--nodes_b', '1']
            )))
            print(ignore, mess_up)
        ignore_cross_messup_3d.append(ignore_row)

    # data.plot_3d_mesh(ignore_cross_messup_3d)
    data.plot_3d_bars(ignore_cross_messup_3d,
                      "ignore_odds (x0.1)", "mess_up_odds (x0.1)", "# simulations",
                      "string consensus, 30 runs, [5, 6, 7] length\n10-node one-to-all bipartite graph")
    plt.savefig('figures/string_mistakes_onetoall_3D.png')


def string_word_length():
    string_lengths = [
        "[0, 1, 2]",
        "[1, 2, 3]",
        "[2, 3, 4]",
        "[3, 4, 5]",
        "[4, 5, 6]",
        "[5, 6, 7]",
        "[6, 7, 8]",
        "[7, 8, 9]",
        "[8, 9, 10]",
        "[9, 10, 11]",
        "[10, 11, 12]"
    ]
    sims = []
    for length_set in string_lengths:
        sims.append(float(call_command(
            ['python', 'levenshtein_driver.py',
             'regular', '100', length_set,
             '0.1', '0.1', '--nodes_a', '20', '--neighbor', '4']
        )))
        print(length_set)

    data.plot_2d_bars(sims,
                      'length of word ([x, x+1, x+2])', '# simulations', 'string consensus, 100 runs, variable length\n4-neighbor regular graph')
    plt.savefig('figures/string_word_length.png')


def alphabet_length():
    # eng alphabet
    eng_result = float(call_command(
        ['python', 'levenshtein_driver.py',
         'small_world', '500', '[5, 6, 7]',
         '0.1', '0.1', '--nodes_a', '20', '--neighbor', '4', '--rewire', '0.25']
    ))
    print(eng_result)

    # jpn hiragana
    jpn_result = float(call_command(
        ['python', 'levenshtein_driver.py',
         'small_world', '500', '[5, 6, 7]',
         '0.1', '0.1', '--nodes_a', '20', '--neighbor', '4', '--rewire', '0.25', '-hiragana']
    ))
    print(jpn_result)

    data.plot_2d_bars([eng_result, jpn_result],
                      '0 = english: 26 letters, 1 = japanese: 45 characters', '# simulations', 'string consensus, 500 runs,[5, 6, 7] length\n4-neighbor 25% rewire small-world graph')
    plt.savefig('figures/alphabet_length.png')


def _show_current_state(graph, words, label, pos=None, filename=None):
    plt.figure()
    graph = data.associate_strings_with_nodes(graph, words)
    plt.title(label)
    ll.draw_graph(graph, abort=True, pos=pos)
    if filename:
        plt.savefig(f"figures/accent_preview_{label}.png")


def accent_w_preview():
    r = (call_command(
        ['python', 'levenshtein_driver.py',
         'barbells', '1', '[5, 6, 7]',
         '0.1', '0.1', '--nodes_a', '10', '--nodes_b', '10',
         '--len_bar', '5', '--num_bars', '1', '-show_graph',
         '-debug', '--preview', '[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]']
    ))
    dict_of_previews = (data.parse_levenshtein_preview_string_to_dictionary(r))

    graph, _ = ll.create_2bell_bars_graph_w_pop_string(
        10, 10, 5, 1, [5, 6, 7])
    pos = nx.kamada_kawai_layout(graph)
    for key in list(dict_of_previews.keys()):
        _show_current_state(
            graph, dict_of_previews[key], label=key, pos=pos, filename=key)


def metric_effects(nodes: int, runs: int):
    iterations = []
    clusterings = []
    shortest_paths = []
    for rewire_odds in range(1, 10):
        r = call_command(
            ['python', 'graph_driver.py',
             'small_world', str(runs),
             '--nodes_a', str(nodes),
             '--neighbor', '4', '--rewire', str(rewire_odds * 0.1),
             '-metrics'])
        r = data.parse_string_to_dict(r)

        print(rewire_odds * 0.1)

        # extend lists w/ new data points
        iterations.extend(r['iterations'])
        clusterings.extend(r['clusterings'])
        shortest_paths.extend(r['shortest_paths'])

    # plot iterations, clusterings, shortest paths
    # print(iterations, clusterings, shortest_paths)

    data.plot_3d_points_with_regression(
        independent_1=shortest_paths, independent_2=clusterings, dependent=iterations,
        i1_label='avg. shortest path length', i2_label='avg. clustering coefficient', dependent_label='iterations',
        title=f'proportion consensus, 9x{runs} runs, 10 symbols\n' +
        f'{nodes} node 4-neighbor variable-rewire small-world graph')

    plt.savefig(f'figures/metric_effects_{str(nodes)}node.png')


def simulate_nicaragua():
    # replicates at home w/ 9 family members
    home_command = ['python', 'graph_driver.py',
                    'bipartite', '50',
                    '--nodes_a', '1', '--nodes_b', '9',
                    '-show_graph']

    # replicates at school w/ 30 students
    school_command = ['python', 'graph_driver.py',
                      'complete', '50',
                      '--nodes_a', '30',
                      '-show_graph']

    # home_r = (call_command(home_command))
    # print(home_r)

    school_r = call_command(school_command)
    print(school_r)


# simulate_nicaragua()
x = [449, 1063, 601, 412, 662, 282, 856, 646, 876, 229, 499, 1131, 854, 692, 467, 414, 876, 525, 332, 517, 491, 871, 448, 625, 193,
     514, 235, 1070, 1153, 471, 446, 614, 867, 250, 1919, 881, 472, 297, 329, 1982, 516, 1316, 542, 1298, 526, 1117, 399, 289, 717, 306]
print(sum(x)/len(x))
