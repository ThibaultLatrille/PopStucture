from common import *


# Function to simulate population structure
@nb.jit(nopython=True)
def simulate_migration(population_size: int, mutation_rate: float, num_generations: int,
                       num_subpopulations: int, migration_rate: float) -> (float, float):
    """
    Simulate the fixation of neutral mutations in a structured population.
    """
    subpopulation_size = population_size // num_subpopulations

    subpopulations = np.zeros((num_subpopulations, subpopulation_size))  # Initialize subpopulations
    substitutions = 0
    gen_diversity = 0
    for generation in range(num_generations):

        for i in range(num_subpopulations):
            # Introduce mutations in each subpopulation
            mutations = np.random.rand(subpopulation_size) < mutation_rate
            subpopulations[i][mutations] = 1

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
        if np.all(subpopulations == 1):
            substitutions += 1
            subpopulations[:] = 0  # Reset to ancestral state for next substitution
        gen_diversity += np.sum(subpopulations) / (num_subpopulations * subpopulation_size)
    return substitutions / num_generations, gen_diversity / num_generations


NUM_SUBPOPULATIONS = 5  # Number of subpopulations
MIGRATION_RATE = 0.01  # Rate of migration between subpopulations
run_simulations(simulate_migration, {"num_subpopulations": NUM_SUBPOPULATIONS, "migration_rate": MIGRATION_RATE},
                "migration", NUM_GENERATIONS=int(1e7), MUTATION_RATE=1e-4, MUTATION_RATES=np.logspace(-5, -3, 10))
