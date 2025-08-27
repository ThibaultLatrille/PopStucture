# PopStructure

Evolution of neutral mutations in structured populations and compares them to panmictic (well-mixed) populations.

## Project Structure

```
PopStucture/
├── scripts/           # Python simulation scripts
│   ├── common.py      # Shared functions and simulation framework
│   ├── harem.py       # Harem mating system simulation
│   ├── migration.py   # Migration between subpopulations simulation
│   ├── pfix_harem.py  # Fixation probability for harem system
│   └── pfix_migration.py # Fixation probability for migration system
├── figures/           # Generated plots and results
└── README.md         # This file
```

## Dependencies

- numPy, matplotlib and numba (JIT compilation).

## Running the Simulations

Each script can be run independently:

```bash
python scripts/harem.py
python scripts/migration.py
python scripts/pfix_harem.py
python scripts/pfix_migration.py
```

## Scripts Overview

### `common.py`
The core module containing shared functions and the simulation framework:

- **`simulate_population()`**: Simulates neutral mutation evolution in a single panmictic population
- **`run_simulations()`**: Runs multiple replicate simulations (10 by default) across different mutation rates and population sizes, computing means and standard errors for error bar plotting
- **`run_simulations_pfix()`**: Similar to above but specifically for fixation probability calculations
- **`plot_results()`**: Creates plots with error bars comparing structured vs. panmictic populations
- **`simulate_pfix_panmixie()`**: Calculates fixation probability in panmictic populations

### `harem.py`
Simulates a **harem mating system** where a small proportion of males (1% by default) contribute disproportionately to reproduction:

- Half the next generation comes from the small male pool, half from females
- Compares substitution rates and genetic diversity to neutral expectations
- Generates plots for both mutation rate and population size effects

### `migration.py`
Simulates **structured populations with migration** between subpopulations:

- Divides the population into 5 subpopulations by default
- Models genetic drift within each subpopulation
- Includes Poisson-distributed migration between subpopulations (1% migration rate)
- Examines how population structure affects evolutionary dynamics

### `pfix_harem.py`
Calculates **fixation probabilities** specifically for the harem mating system:

- Introduces single mutations and tracks their fate
- Runs 100,000 independent mutations per simulation
- Compares fixation probability to the neutral expectation of 1/N

### `pfix_migration.py`
Calculates **fixation probabilities** for the migration model:

- Similar to harem version but for structured populations with migration
- Tracks single mutations from introduction to fixation or loss
- Compares structured vs. panmictic fixation probabilities

## Simulation Parameters

### Default Values
- **Population size**: 100 
- **Mutation rate**: 10⁻⁴ 
- **Number of generations**: 10⁷ (for substitution rate simulations)
- **Number of mutations**: 10⁵ (for fixation probability simulations)
- **Number of replicates**: 10 per parameter combination

### Model-Specific Parameters
- **Harem model**: 1% male proportion
- **Migration model**: 5 subpopulations, 1% migration rate between them

## Output

The simulations generate several types of plots in the `figures/` directory:

1. **Substitution Rate vs. Mutation Rate**: How the rate of neutral substitutions varies with mutation rate
2. **Substitution Rate vs. Population Size**: Effect of population size on substitution rates
3. **Genetic Diversity vs. Mutation Rate**: Relationship between mutation rate and standing genetic variation
4. **Genetic Diversity vs. Population Size**: How population size affects genetic diversity
5. **Fixation Probability (Pfix) vs. Population Size**: Probability that a new mutation will fix

Each plot compares:
- **Single Population** (panmictic): Well-mixed population serving as control
- **Structured Population**: Either harem or migration model
- **Neutral Expectation**: Theoretical prediction

