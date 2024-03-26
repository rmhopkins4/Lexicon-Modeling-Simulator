from homesign_simulator import repeat_sims as rs

# Set number of simulations to run
num_runs = 10

# Set population
population_count = 5

# Set number of symbols per speaker
symbol_count = 30


# Run simulation w/ language model
result = rs(True, num_runs, population_count, symbol_count)
print(
    f'Took {result} steps using language model to conventionalize {symbol_count} symbols with a population of {population_count}.')

# Run simulation w/ homesign model
result = rs(False, num_runs, population_count, symbol_count)
print(
    f'Took {result} steps using homesign model to conventionalize {symbol_count} symbols with a population of {population_count}.')
