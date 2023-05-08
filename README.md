README

This program simulates the evolution of communication systems in populations. The simulation consists of a group of individuals, each with a probability distribution over a set of symbols. The program uses numpy and random libraries for calculations and generating random numbers.
The main components of the program are as follows:

fill_array(length): This function generates an array of random normalized doubles with a specified length.

build_population(population_count, symbol_count): This function builds a population of individuals with a specified count of symbols.

interact(speaker, listener): This function models the interaction between two individuals - one acting as a speaker, the other as a listener. The listener learns from the speaker's use of symbols and adjusts its own probability distribution accordingly.

generate_interaction_matrix(is_language, population_count): This function creates a matrix of probabilities for the interactions between the individuals in the population.

run_simulation(population, interaction_matrix): This function runs the simulation for one step by randomly selecting pairs of individuals to interact with each other.

repeat_sims(is_language=True, num_runs=10, population_count=5, symbol_count=10): This function repeats the simulation for a specified number of runs, with a specified population count and symbol count, and calculates the average number of steps required for convergence.

The variable learn_coefficient is used to set the learning rate for the listeners.

To use the program, simply import the required libraries and call the functions as needed, with appropriate parameters.

Driver file shows example usage.
