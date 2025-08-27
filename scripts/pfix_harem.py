from common import *


# Function to simulate population structure
@nb.jit(nopython=True)
def simulate_pfix_harem(population_size: int, nbr_mutations: int, male_proportion: float) -> float:
    """
    Simulate the fixation of neutral mutations in a structured population.
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
    substitutions = 0

    for mutation in range(nbr_mutations):
        # Initialize population
        alleles = np.zeros(population_size)  # All individuals start with the same allele (0)
        # Mutate one individual
        mutation = np.random.randint(population_size)
        alleles[mutation] = 1  # Introduce a new allele upon mutation
        assert np.sum(alleles) == 1
        while True:
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

            # Check for fixation
            tot = np.sum(alleles)
            if tot == len(alleles):
                substitutions += 1
                break
            elif tot == 0:
                break

    return substitutions / nbr_mutations


MALE_PROPORTION = 0.01  # Proportion of males in the population
run_simulations_pfix(simulate_pfix_harem,
                     {"male_proportion": MALE_PROPORTION},
                     "harem", NBR_MUTATIONS=int(1e5))
