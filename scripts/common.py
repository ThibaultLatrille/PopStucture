import numpy as np
import matplotlib.pyplot as plt
import numba as nb


# Function to simulate a single population
@nb.jit(nopython=True)
def simulate_population(population_size: int, mutation_rate: float, num_generations: int) -> (float, float):
    """
    Simulate the fixation of neutral mutations in a single population.
    """
    alleles = np.zeros(population_size)  # All individuals start with the same allele (0)
    substitutions = 0
    gen_diversity = 0

    for generation in range(num_generations):
        # Mutate individuals
        mutations = np.random.rand(population_size) < mutation_rate
        alleles[mutations] = 1  # Introduce a new allele (1) upon mutation

        # Sample alleles to form the next generation (genetic drift)
        if np.sum(alleles) > 0:
            alleles = np.random.choice(alleles, size=population_size, replace=True)

        # Check for fixation
        if np.all(alleles == 1):
            substitutions += 1
            alleles[:] = 0  # Reset to ancestral state for next substitution

        gen_diversity += np.sum(alleles) / population_size
    return substitutions / num_generations, gen_diversity / num_generations


def plot_results(x_list, single_list, structured_list, neutral_list, x_label, y_label, title, filename,
                 single_errors=None, structured_errors=None, xscale="linear", yscale="linear"):
    # Plot the results
    plt.figure(figsize=(10, 6))
    if single_errors is not None:
        plt.errorbar(x_list, single_list, yerr=single_errors, label="Single Population", marker='o', capsize=5)
    else:
        plt.plot(x_list, single_list, label="Single Population", marker='o')

    if structured_errors is not None:
        plt.errorbar(x_list, structured_list, yerr=structured_errors, label="Structured Population", marker='o', capsize=5)
    else:
        plt.plot(x_list, structured_list, label="Structured Population", marker='o')

    plt.plot(x_list, neutral_list, label="Neutral Expectation", linestyle='--', color='gray')
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close("all")
    plt.clf()


def run_simulations(simulate_population_structure, population_structure_params, name,
                    POPULATION_SIZE=100, POPULATION_SIZES=np.logspace(2, 3, 8),
                    MUTATION_RATE=1e-4, MUTATION_RATES=np.logspace(-5, -3, 10),
                    NUM_GENERATIONS=5000000, NUM_REPLICATES=10):
    print(f"Running simulations for {NUM_GENERATIONS} generations with {NUM_REPLICATES} replicates")

    # Simulate and plot results for different mutation rates
    q_panmixie_means, q_structured_means = [], []
    theta_panmixie_means, theta_structured_means = [], []
    q_panmixie_errors, q_structured_errors = [], []
    theta_panmixie_errors, theta_structured_errors = [], []

    for mu in MUTATION_RATES:
        print(f"Simulating mutation rate: {mu}")

        # Run multiple replicates for panmixie
        q_panmixie_reps, theta_panmixie_reps = [], []
        for rep in range(NUM_REPLICATES):
            rate, diversity = simulate_population(POPULATION_SIZE, mu, NUM_GENERATIONS)
            q_panmixie_reps.append(rate)
            theta_panmixie_reps.append(diversity)

        # Run multiple replicates for structured
        q_structured_reps, theta_structured_reps = [], []
        for rep in range(NUM_REPLICATES):
            rate, diversity = simulate_population_structure(POPULATION_SIZE, mu, NUM_GENERATIONS,
                                                          **population_structure_params)
            q_structured_reps.append(rate)
            theta_structured_reps.append(diversity)

        # Compute means and standard errors
        q_panmixie_means.append(np.mean(q_panmixie_reps))
        theta_panmixie_means.append(np.mean(theta_panmixie_reps))
        q_structured_means.append(np.mean(q_structured_reps))
        theta_structured_means.append(np.mean(theta_structured_reps))

        q_panmixie_errors.append(np.std(q_panmixie_reps) / np.sqrt(NUM_REPLICATES))
        theta_panmixie_errors.append(np.std(theta_panmixie_reps) / np.sqrt(NUM_REPLICATES))
        q_structured_errors.append(np.std(q_structured_reps) / np.sqrt(NUM_REPLICATES))
        theta_structured_errors.append(np.std(theta_structured_reps) / np.sqrt(NUM_REPLICATES))

    plot_results(MUTATION_RATES, q_panmixie_means, q_structured_means, MUTATION_RATES,
                 "Mutation Rate", "Substitution Rate", "Substitution Rate as a Function of Mutation Rate",
                 f"figures/SubRate_MutRate_{name}.pdf", q_panmixie_errors, q_structured_errors,
                 xscale="log", yscale="log")
    plot_results(MUTATION_RATES, theta_panmixie_means, theta_structured_means, 2 * MUTATION_RATES * POPULATION_SIZE,
                 "Mutation Rate", "Genetic Diversity", "Genetic Diversity as a Function of Mutation Rate",
                 f"figures/Diversity_MutRate_{name}.pdf", theta_panmixie_errors, theta_structured_errors,
                 xscale="log", yscale="log")

    # Simulate and plot results for different population sizes
    q_panmixie_means, q_structured_means = [], []
    theta_panmixie_means, theta_structured_means = [], []
    q_panmixie_errors, q_structured_errors = [], []
    theta_panmixie_errors, theta_structured_errors = [], []

    for ne in map(int, POPULATION_SIZES):
        print(f"Simulating population size: {ne}")

        # Run multiple replicates for panmixie
        q_panmixie_reps, theta_panmixie_reps = [], []
        for rep in range(NUM_REPLICATES):
            rate, diversity = simulate_population(ne, MUTATION_RATE, NUM_GENERATIONS)
            q_panmixie_reps.append(rate)
            theta_panmixie_reps.append(diversity)

        # Run multiple replicates for structured
        q_structured_reps, theta_structured_reps = [], []
        for rep in range(NUM_REPLICATES):
            rate, diversity = simulate_population_structure(ne, MUTATION_RATE, NUM_GENERATIONS,
                                                          **population_structure_params)
            q_structured_reps.append(rate)
            theta_structured_reps.append(diversity)

        # Compute means and standard errors
        q_panmixie_means.append(np.mean(q_panmixie_reps))
        theta_panmixie_means.append(np.mean(theta_panmixie_reps))
        q_structured_means.append(np.mean(q_structured_reps))
        theta_structured_means.append(np.mean(theta_structured_reps))

        q_panmixie_errors.append(np.std(q_panmixie_reps) / np.sqrt(NUM_REPLICATES))
        theta_panmixie_errors.append(np.std(theta_panmixie_reps) / np.sqrt(NUM_REPLICATES))
        q_structured_errors.append(np.std(q_structured_reps) / np.sqrt(NUM_REPLICATES))
        theta_structured_errors.append(np.std(theta_structured_reps) / np.sqrt(NUM_REPLICATES))

    plot_results(POPULATION_SIZES, q_panmixie_means, q_structured_means, np.ones(len(POPULATION_SIZES)) * MUTATION_RATE,
                 "Population Size", "Substitution Rate", "Substitution Rate as a Function of Population Size",
                 f"figures/SubRate_PopSize_{name}.pdf", q_panmixie_errors, q_structured_errors,
                 xscale="log", yscale="log")
    plot_results(POPULATION_SIZES, theta_panmixie_means, theta_structured_means, 2 * MUTATION_RATE * POPULATION_SIZES,
                 "Population Size", "Genetic Diversity", "Genetic Diversity as a Function of Population Size",
                 f"figures/Diversity_PopSize_{name}.pdf", theta_panmixie_errors, theta_structured_errors,
                 xscale="log", yscale="log")


# Function to pfix a single population
@nb.jit(nopython=True)
def simulate_pfix_panmixie(population_size: int, nbr_mutations: int) -> float:
    """
    pfix the fixation of neutral mutations in a single population.
    """
    substitutions = 0
    for mutation in range(nbr_mutations):
        # Initialize population
        alleles = np.zeros(population_size)  # All individuals start with the same allele (0)
        # Mutate one individual
        mutation = np.random.randint(population_size)
        alleles[mutation] = 1  # Introduce a new allele upon mutation
        assert np.sum(alleles) == 1
        while True:
            # Sample alleles to form the next generation (genetic drift)
            alleles = np.random.choice(alleles, size=population_size, replace=True)

            # Check for fixation
            tot = np.sum(alleles)
            if tot == len(alleles):
                substitutions += 1
                break
            elif tot == 0:
                break

    return substitutions / nbr_mutations


def run_simulations_pfix(pfix_population_structure, population_structure_params, name,
                         POPULATION_SIZES=np.logspace(2, 3, 8),
                         NBR_MUTATIONS=10000, NUM_REPLICATES=10):
    print(f"Running simulations for {NBR_MUTATIONS} mutations with {NUM_REPLICATES} replicates")

    # pfix and plot results for different population sizes
    pfix_panmixie_means, pfix_structured_means = [], []
    pfix_panmixie_errors, pfix_structured_errors = [], []

    for ne in map(int, POPULATION_SIZES):
        print(f"Simulating population size: {ne}")

        # Run multiple replicates for panmixie
        pfix_panmixie_reps = []
        for rep in range(NUM_REPLICATES):
            pfix_panmixie_reps.append(simulate_pfix_panmixie(ne, NBR_MUTATIONS))

        # Run multiple replicates for structured
        pfix_structured_reps = []
        for rep in range(NUM_REPLICATES):
            pfix_structured_reps.append(pfix_population_structure(ne, NBR_MUTATIONS, **population_structure_params))

        # Compute means and standard errors
        pfix_panmixie_means.append(np.mean(pfix_panmixie_reps))
        pfix_structured_means.append(np.mean(pfix_structured_reps))
        pfix_panmixie_errors.append(np.std(pfix_panmixie_reps) / np.sqrt(NUM_REPLICATES))
        pfix_structured_errors.append(np.std(pfix_structured_reps) / np.sqrt(NUM_REPLICATES))

    plot_results(POPULATION_SIZES, pfix_panmixie_means, pfix_structured_means, 1 / POPULATION_SIZES,
                 "Population Size", "Pfix", "Pfix as a Function of Population Size",
                 f"figures/Pfix_PopSize_{name}.pdf", pfix_panmixie_errors, pfix_structured_errors,
                 xscale="log", yscale="log")
