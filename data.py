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
