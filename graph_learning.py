import networkx as nx
import matplotlib.pyplot as plt

RANDOM_SEED = 118755212


def create_regular_graph_w_pop(n, k, num_symbols):
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(range(n))

    # Connect each node to k neighbors (k/2 on each side)
    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbor1 = (i + j) % n
            neighbor2 = (i - j) % n
            G.add_edge(i, neighbor1)
            G.add_edge(i, neighbor2)

    # return G


def create_lopsided_barbell_graph_w_pop(n, m, k, num_symbols):
    # Create first complete
    bell1 = nx.complete_graph(n)

    for i in range(k):
        bell1.add_node(n+i)
        bell1.add_edge(n+i-1, n+i)

    bell2 = nx.complete_graph(m)

    # Combine the two bells using disjoint_union
    barbell_graph = nx.disjoint_union(bell1, bell2)

    barbell_graph.add_edge(n+k-1, n+k)

    return barbell_graph

# random drift (together) after interactions? more impact on barbell-type graphs, accents
#   depends on how i model current state, difficult to do w/ homesign method, which in all likelihood i will keep

# complete, random initial opinions, how long before consensus for different sizes?
# barbell, only two distinct initial agreements, tweak sizes (relative, total) to predict which wins and how long it takes?
# bipartite, compare w/ barbell
# lollipop, how long to fully agree depending on length of stick and size of candy?
# small world vs regular, how much does a small set of extra connections help w/ consensus time?
#

# barabási–albert?

# Function to create a regular graph with n nodes and k neighbors


g = create_lopsided_barbell_graph_w_pop(10, 20, 0)
a = nx.to_numpy_array(g)
print(a)
nx.draw(g, with_labels=True, node_color='skyblue', node_size=100)
plt.show()
