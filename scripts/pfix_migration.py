from common import *


# Function to simulate population structure
@nb.jit(nopython=True)
def simulate_pfix_migration(population_size: int, nbr_mutations: int, num_subpopulations: int,
                            migration_rate: float) -> float:
    """
    Simulate the fixation of neutral mutations in a structured population.
    """
    subpopulation_size = population_size // num_subpopulations

    substitutions = 0
    for mutation in range(nbr_mutations):
        subpopulations = np.zeros((num_subpopulations, subpopulation_size))  # Initialize subpopulations
        # Mutate one individual in one subpopulation
        rand_ind = np.random.randint(subpopulation_size)
        rand_pop = np.random.randint(num_subpopulations)
        subpopulations[rand_pop][rand_ind] = 1
        assert np.sum(subpopulations) == 1
        pop_size = num_subpopulations * subpopulation_size
        while True:
            for i in range(num_subpopulations):
                # Genetic drift within each subpopulation
                if np.sum(subpopulations[i]) > 0:
                    subpopulations[i] = np.random.choice(subpopulations[i], size=subpopulation_size, replace=True)

            # Migration between subpopulations
            for i in range(num_subpopulations):
                for j in range(num_subpopulations):
                    if i != j:
                        num_migrants = np.random.poisson(migration_rate * subpopulation_size)
                        migrants = np.random.choice(subpopulations[i], size=num_migrants, replace=False)
                        subpopulations[j][:num_migrants] = migrants

            # Check for fixation across the entire population
            tot = np.sum(subpopulations)
            # Total population size as the len of the bidimensional array
            if tot == pop_size:
                substitutions += 1
                break
            elif tot == 0:
                break
    return substitutions / nbr_mutations


NUM_SUBPOPULATIONS = 5  # Number of subpopulations
MIGRATION_RATE = 0.01  # Rate of migration between subpopulations
run_simulations_pfix(simulate_pfix_migration,
                     {"num_subpopulations": NUM_SUBPOPULATIONS, "migration_rate": MIGRATION_RATE},
                     "migration", NBR_MUTATIONS=int(1e5))
