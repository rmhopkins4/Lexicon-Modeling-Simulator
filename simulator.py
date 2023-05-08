import random
import numpy as np


# Fill array with random normalized doubles
def fill_array(length):
    array = np.random.random(length)
    return array / np.sum(array)


# Build population
def build_population(population_count, symbol_count):
    population = np.random.rand(population_count, symbol_count)
    return np.divide(population, np.sum(population, axis=1, keepdims=True))


# Speaker condenses probabilities, listener learns
def interact(speaker, listener):
    performed_symbol = np.random.choice(len(speaker), p=speaker)
    # non-performed symbols become less common
    listener[np.arange(len(listener) != performed_symbol)
             ] *= (1-learn_coefficient)
    # performed symbol becomes more common
    listener[performed_symbol] += learn_coefficient * (1-learn_coefficient)
    listener /= np.sum(listener)


# Creates matrix of probabilities of interaction
def generate_interaction_matrix(is_language, population_count):
    if is_language:
        matrix = np.full((population_count, population_count),
                         1 / (population_count-1))
        np.fill_diagonal(matrix, 0)
    else:
        matrix = np.zeros((population_count, population_count))
        matrix[0] = 1 / (population_count-1)
        matrix[1:, 0] = 1
        matrix[0][0] = 0
    return matrix


# Runs simulation one step
def run_simulation(population, interaction_matrix):
    for i in range(len(interaction_matrix)):
        for j in range(len(interaction_matrix[i])):
            rand = random.random()
            if rand < interaction_matrix[i][j]:
                interact(population[i], population[j])
    return np.allclose(population, population[0], atol=1e-9)


# Runs simulation repeatedly, finding average number of steps
def repeat_sims(is_langauge=True, num_runs=10, population_count=5, symbol_count=10):
    interaction_matrix = generate_interaction_matrix(
        is_language=is_langauge, population_count=population_count)
    total_sum = 0
    for i in range(num_runs):
        # generate new population
        population = build_population(population_count, symbol_count)
        sum = 0
        while not run_simulation(population=population,
                                 interaction_matrix=interaction_matrix):
            total_sum += 1
    return total_sum / num_runs  # average number of steps


# Set learn coefficient
learn_coefficient = 0.02
