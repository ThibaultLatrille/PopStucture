from common import *


# Function to simulate population structure with a harem model
@nb.jit(nopython=True)
def simulate_harem(population_size: int, mutation_rate: float, num_generations: int,
                   male_proportion: float) -> (float, float):
    """
    Simulate the fixation of neutral mutations in a structured population with a harem structure.
    """
    # if population_size not even add one
    if population_size % 2 != 0:
        population_size += 1
    num_males = int(male_proportion * population_size)
    assert num_males > 0, "Number of males must be greater than 0"
    half_population = population_size // 2
    assert half_population * 2 == population_size, "population size must be even"
    num_females = population_size - num_males
    assert num_females > 0, "Number of females must be greater than 0"
    alleles = np.zeros(population_size)  # Initialize alleles
    substitutions = 0
    gen_diversity = 0
    for generation in range(num_generations):
        # Introduce mutations in each population
        mutations = np.random.rand(population_size) < mutation_rate
        alleles[mutations] = 1

        # Genetic drift within each population (harem structure)
        males = alleles[:num_males]
        females = alleles[num_males:]

        # Males contribute disproportionately to the next generation
        # Half the population is from males, half from females
        next_gen_m = np.random.choice(males, size=half_population, replace=True)
        next_gen_f = np.random.choice(females, size=half_population, replace=True)
        # Combine the two groups to form the next generation, and shuffle
        next_gen = np.concatenate((next_gen_m, next_gen_f))
        np.random.shuffle(next_gen)
        alleles = next_gen

        # Check for fixation across the entire population
        if np.all(alleles == 1):
            substitutions += 1
            alleles[:] = 0  # Reset to ancestral state for next substitution

        gen_diversity += np.sum(alleles) / population_size

    return substitutions / num_generations, gen_diversity / num_generations


MALE_PROPORTION = 0.01  # Proportion of males in the population
run_simulations(simulate_harem, {"male_proportion": MALE_PROPORTION},
                "harem", NUM_GENERATIONS=int(1e7), MUTATION_RATE=1e-4, MUTATION_RATES=np.logspace(-5, -3, 10))
