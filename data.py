from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import ast
import networkx as nx
from Levenshtein import distance as levenshtein_distance
from Levenshtein import distance as levenshtein_distance  # Import distance function
from Levenshtein import distance
import distance
from sklearn.cluster import AffinityPropagation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pylab as plt

import random


def plot_3d_mesh(data, x_label, y_label, z_label, title):
    array = np.array(data)
    # Create meshgrid for X and Y axes
    x, y = np.meshgrid(np.arange(array.shape[0]), np.arange(array.shape[1]))

    # Create figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    ax.plot_surface(x, y, array, cmap='viridis')

    # Set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.title(title)

    # plt.show()


def plot_3d_bars(data, x_label, y_label, z_label, title):
    # Create figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define colormap
    cmap = cm.get_cmap('viridis')

    # Normalize data for colormap
    min_val = min(min(row) for row in data)
    max_val = max(max(row) for row in data)
    norm = plt.Normalize(min_val, max_val)

    # Loop through the data and plot each bar
    for i in range(len(data)):
        for j in range(len(data[i])):
            color = cmap(norm(data[i][j]))
            ax.bar3d(i, j, 0, 1, 1, data[i][j], color=color, alpha=1)

    # Set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.title(title)

    # plt.show()


def plot_3d_points(dependent, independent_1, independent_2, i1_label, i2_label, dependent_label, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(independent_1, independent_2, dependent, c='r', marker='o')
    ax.set_xlabel(i1_label)
    ax.set_ylabel(i2_label)
    ax.set_zlabel(dependent_label)
    plt.title(title)


def plot_3d_points_with_regression(independent_1, independent_2, dependent, i1_label, i2_label, dependent_label, title):
    # Create figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points in 3D
    ax.scatter(independent_1, independent_2, dependent, c='r', marker='o')

    # Fit a linear regression model
    model = LinearRegression()
    X = np.column_stack((independent_1, independent_2))
    model.fit(X, dependent)

    # Predict values using the model
    predicted_dependent = model.predict(X)

    # Plot the regression line
    ax.plot_trisurf(independent_1, independent_2,
                    predicted_dependent, color='blue', alpha=0.5)

    # Set labels
    ax.set_xlabel(i1_label)
    ax.set_ylabel(i2_label)
    ax.set_zlabel(dependent_label)

    # Calculate R^2
    r_squared = r2_score(dependent, predicted_dependent)
    # print("R^2:", r_squared)

    plt.title(f"{title}\nR²:{r_squared}")


def plot_2d_bars(data, x_label, y_label, title, max_value=None):
    plt.figure()
    norm = plt.Normalize(min(data), max(data))
    cmap = plt.get_cmap('viridis')
    colors = cmap(norm(data))

    plt.bar(range(len(data)), data, color=colors)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.gca().xaxis.set_major_locator(
        plt.MaxNLocator(integer=True))  # Set integer ticks on x-axis

    if max_value is not None:
        plt.ylim(0, max_value)


def plot_22d_bars(data1, data2, x_label, y_label, title1, title2, max_value=None):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 5))  # Adjust figsize as needed

    # Calculate averages
    avg1 = np.mean(data1)
    avg2 = np.mean(data2)

    norm1 = plt.Normalize(min(data1), max(data1))
    cmap1 = plt.get_cmap('viridis')
    colors1 = cmap1(norm1(data1))

    ax1.bar(range(len(data1)), data1, color=colors1)
    ax1.axhline(y=avg1, color='r', linestyle='--',
                label=f'Average: {avg1:.2f}')  # Add horizontal line
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title1)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(
        integer=True))  # Set integer ticks on x-axis

    norm2 = plt.Normalize(min(data2), max(data2))
    cmap2 = plt.get_cmap('viridis')
    colors2 = cmap2(norm2(data2))

    ax2.bar(range(len(data2)), data2, color=colors2)
    ax2.axhline(y=avg2, color='r', linestyle='--',
                label=f'Average: {avg2:.2f}')  # Add horizontal line
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_title(title2)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(
        integer=True))  # Set integer ticks on x-axis

    if max_value is not None:
        ax1.set_ylim(0, max_value)
        ax2.set_ylim(0, max_value)

    ax1.legend()  # Add legend to first subplot
    ax2.legend()  # Add legend to second subplot

    plt.tight_layout()  # Adjust subplots to prevent overlap


def associate_strings_with_nodes(graph, strings):

    for i, string in enumerate(strings):
        if i in graph.nodes:
            graph.nodes[i]['label'] = string
    return graph


def parse_levenshtein_preview_string_to_dictionary(s):
    result = {}
    for line in s.split('\n'):
        if line.strip() and line.startswith('P'):  # Skip empty lines and lines not starting with 'P'
            parts = line.split(':')
            index = int(parts[0].split()[1])  # Extract the integer after 'P'
            words = parts[1].strip().strip("[]").replace("'", "").split(', ')
            result[index] = words
    return result


def parse_string_to_dict(string):
    try:
        return ast.literal_eval(string)
    except:
        print("Error: Invalid string representation of dictionary")
        return None


# Define the strings at different time points
time_points = [10, 20, 30, 40, 50, 100, 150, 200]
strings = [
    ['tjjiw', 'aeziz', 'etdos', 'atzupw', 'tduiw', 'ejtup', 'aedjiz', 'btezvpv', 'ejztwe', 'bejipp', 'gsfzl', 'gsfmp', 'gsamq',
        'igxzr', 'ispqm', 'gxzamr', 'ipemm', 'ypsfmp', 'gxzalm', 'wsfmq', 'biezvpv', 'yieiue', 'yioiue', 'yioiue', 'yiosue'],
    ['btbjjpv', 'tioz', 'botpv', 'boipv', 'btjdov', 'bopv', 'beoive', 'bjoiuv', 'tbooive', 'tjdov', 'wszfmq', 'isufmq', 'wguzmq',
        'wguszmq', 'wiuamq', 'wguamq', 'wsuzmq', 'ioiuvmq', 'wguzm', 'wgszmq', 'yioiuv', 'yioiuv', 'yioiuv', 'yioiuv', 'yioiuv'],
    ['bjioiv', 'bjioiv', 'tbjiov', 'tbjiov', 'bjiov', 'bjiov', 'tbjioive', 'bjiov', 'bjiov', 'tbjioiv', 'wguzmq', 'wgozmq', 'wgozmq',
        'wgozmq', 'wgozmq', 'wiuzmq', 'wiuzmq', 'wiozmv', 'wiozmq', 'wgozmq', 'yiozuv', 'yiozuv', 'yiozuv', 'yiozuv', 'yiozuv'],
    ['bjioiv', 'bjioiv', 'bjioie', 'tbjiozv', 'bjioiv', 'tbjioiv', 'tbjioiv', 'bjioziv', 'bjioiv', 'tbjioie', 'wiozmq', 'wiozmq',
        'wiuzmq', 'wiozmq', 'wiuzmq', 'wiuzuq', 'wiozuq', 'wiozuv', 'wiozmq', 'wiozmq', 'yiozuv', 'yiozuv', 'yiozuv', 'yiozuv', 'yiozuv'],
    ['bjioiv', 'bjioiv', 'bjioiv', 'bjioiv', 'bjioiv', 'bjioiv', 'bjioiv', 'bjioiv', 'bjioiv', 'bjioiv', 'wiuzmv', 'wiuzmq', 'wiuzmq',
        'wiuzmv', 'wiuzuv', 'wiuzuv', 'wiuzmv', 'wiuzuv', 'wiuzmq', 'wiuzmv', 'yiozuv', 'yiozuv', 'yiozuv', 'yiozuv', 'yiozuv'],
    ['yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjioziv', 'yjiouv', 'yjiouv', 'yjioziv', 'yjiouv', 'yjiouv', 'yjiuzuv', 'yiuzuv', 'yiuzuv',
        'yjiuzuv', 'yiuzuv', 'yiuzuv', 'yjiuzuv', 'yjiuzuv', 'yiuzuv', 'yiuzuv', 'yjioziv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiuzuv'],
    ['yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjiozuv', 'yjiozuv', 'yjiouv', 'yjiouv', 'yjiozuv', 'yjiozuv', 'yjiozuv',
        'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv'],
    ['yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjiouv', 'yjiozuv', 'yjiozuv', 'yjiozuv',
        'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv', 'yjiozuv']
]
