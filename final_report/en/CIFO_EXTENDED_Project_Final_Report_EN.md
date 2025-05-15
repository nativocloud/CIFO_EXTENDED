# CIFO EXTENDED Project Comprehensive Final Report

**CIFO Team**

**Date:** May 15, 2025

## Executive Summary

This final report consolidates all phases of the CIFO EXTENDED project, which aimed to optimize sports team selection through the application and analysis of metaheuristic algorithms. The project evolved from single-processor implementations, through code optimizations, exploration of multiprocessing with different numbers of runs, to an in-depth analysis of parameter variation for the most promising algorithms. The main conclusions point to Hill Climbing (specifically with 500 iterations) as the most efficient approach in terms of balancing solution quality, consistency, and execution time. For scenarios requiring the guarantee of the best possible fitness with perfect consistency, an optimized configuration of the Genetic Algorithm (GA_Config_4 with a mutation rate of 0.15) proved superior, albeit at a higher computational cost. This document details the methodology, results, and conclusions of each phase, culminating in recommendations for future work and a diagram of the analysis process employed.

## 1. General Project Introduction

The central challenge of the CIFO EXTENDED project was to develop and evaluate effective computational methods for the complex problem of forming multiple sports teams from a set of available players. This combinatorial optimization problem is characterized by various constraints, including team budgets, number of players per team, and specific positional requirements. The main optimization objective was to minimize the variance of the average skill among the formed teams, thereby promoting competitive balance.

Throughout the project, various metaheuristic algorithms were explored, including Hill Climbing (HC), Simulated Annealing (SA), and Genetic Algorithms (GAs) with a variety of operators. The investigation progressed through several distinct phases:

1.  **Single-Processor Phase:** Initial implementation of algorithms, profiling to identify bottlenecks, and code optimizations (reduction of `deepcopy` and vectorization).
2.  **Multiprocessing Phase - 5 Runs:** Initial evaluation of algorithm performance in parallel to obtain more statistically robust data and select candidates for deeper analysis.
3.  **Multiprocessing Phase - 30 Runs with Promising Algorithms:** More rigorous statistical analysis of selected algorithms (HC, SA, and two GA configurations) with their base parameters.
4.  **Parameter Variation Phase - 30 Runs per Variation:** Detailed investigation of the impact of different hyperparameters (number of iterations for HC; mutation rate, population size, number of generations for selected GAs) on performance.

This report aims to present a consolidated view of the entire process, from initial conception to final conclusions and recommendations.

## 2. Single-Processor Phase: Implementation and Initial Optimizations

### 2.1. Single-Processor Code Architecture (Initial Version)

The code architecture for the single-processor phase was designed modularly to facilitate the implementation, testing, and replacement of different components of the metaheuristic algorithms. The main Python files involved in this architecture are:

*   `solution.py`: Defines the representation of a solution to the problem (an assignment of players to teams) and methods to evaluate its validity and quality (fitness). It also includes methods for generating neighboring solutions or modifying existing solutions.
*   `evolution.py`: Contains the implementations of the main metaheuristic algorithms: Hill Climbing, Simulated Annealing, and the generic Genetic Algorithm.
*   `operators.py`: Implements the different genetic operators used by the Genetic Algorithm, such as selection, crossover, and mutation operators.
*   `main_script_sp.py`: The main script that orchestrates the execution of algorithms, loads player data, configures algorithm parameters, and collects/presents results.

#### 2.1.1. Solution Representation and Data Structures

The rationale behind the choice of solution representation and data structures was simplicity and efficiency for the necessary operations.

*   **Solution Representation (`LeagueSolution` in `solution.py`):** A solution is represented as a list (or, later, a NumPy array) where the index corresponds to a player's ID and the value at that index corresponds to the ID of the team to which that player has been assigned. For example, `assignment[player_id] = team_id`.
    *   **Rationale:** This representation is direct, easy to manipulate, and allows quick access to any player's team. It is also compact.

*   **Player Data:** Player data (ID, name, position, salary, skill) is initially loaded from a CSV file into a convenient data structure (such as a list of dictionaries or a Pandas DataFrame) in the main script. For internal operations of solution and algorithm classes, this data is often passed or converted to formats more optimized for calculation (such as NumPy arrays for salaries, skills, and numerical positions, as was done in the vectorization phase).
    *   **Rationale:** CSV is a common and easy-to-use format for data input. Internally, conversion to NumPy arrays (especially after vectorization) aims to speed up numerical calculations.

*   **Team Structure:** Implicitly, teams are collections of players. The solution representation allows for easy reconstruction of each team's composition by filtering players based on the `team_id` assigned to them.

#### 2.1.2. Implemented Algorithms (Initial, Unoptimized Version)

Three main types of metaheuristic algorithms were selected for this initial phase, due to their popularity and different approaches to exploring the solution space:

1.  **Hill Climbing (HC):**
    *   **Description:** An iterative local search algorithm that starts with an arbitrary solution and tries to find a better solution by moving to a neighbor with better fitness. It terminates when it reaches a local optimum, where no neighbor has better fitness.
    *   **Selection Rationale:** It is one of the simplest optimization algorithms to implement and understand. It serves as a good baseline for comparison with more complex algorithms. It is fast but prone to getting stuck in local optima.

2.  **Simulated Annealing (SA):**
    *   **Description:** A probabilistic optimization technique inspired by the annealing process in metallurgy. It allows moves to worse solutions with a certain probability, which decreases as the algorithm progresses (the "temperature" drops). This helps escape local optima.
    *   **Selection Rationale:** It offers better global exploration capability than Hill Climbing, with the ability to avoid local optima. Its effectiveness depends on the correct parameterization of the cooling schedule.

3.  **Genetic Algorithms (GAs):**
    *   **Description:** Search algorithms inspired by biological evolution. They maintain a population of candidate solutions that evolve over generations through genetic operators such as selection, crossover (recombination), and mutation.
    *   **Selection Rationale:** They are powerful for complex optimization problems and can effectively explore large solution spaces. Their population-based nature allows for maintaining diversity and exploring multiple regions of the search space simultaneously.

#### 2.1.3. Selection of Operators for Testing (Genetic Algorithms)

For Genetic Algorithms, a variety of operators were implemented to allow testing of different evolutionary strategies:

*   **Selection Operators:** Determine which individuals from the current population are chosen to reproduce.
    *   `selection_tournament`: Tournament selection.
    *   `selection_roulette_wheel`: Roulette wheel selection.
    *   `selection_ranking`: Rank-based selection.
    *   `selection_boltzmann`.
*   **Crossover Operators:** Combine the genetic material of two parents to create one or more offspring.
    *   `crossover_one_point`: One-point crossover.
    *   `crossover_uniform`: Uniform crossover.
    *   `_prefer_valid` versions to try to generate valid offspring more directly.
*   **Mutation Operators:** Introduce small random changes in individuals.
    *   `mutation_random_player_team_change`: Randomly changes a player's team.
    *   `mutation_swap_players_between_teams`: Swaps two players between two teams.

The initial selection of these algorithms and operators aimed to cover a spectrum of metaheuristic approaches.

### 2.2. Results of Initial Optimizations (Single-Processor)

After the initial implementation, a profiling and optimization phase was carried out. The two main optimization areas focused on were reducing the use of `deepcopy` and vectorizing critical functions.

#### 2.2.1. Initial Profiling and Bottleneck Identification

The initial execution of the script (`main_script_sp_timing_estimate.py`) revealed that the total time was approximately **85.66 seconds**. `cProfile` highlighted:

*   **Simulated Annealing (SA):** The slowest, consuming about **73.17 seconds**.
*   **`copy.deepcopy`:** Main cause of slowness, called over 23 million times, totaling **53.49 seconds**.
*   **Solution Functions:** `get_random_neighbor`, `fitness`, and `is_valid` also consumed significant time.

#### 2.2.2. Optimization of `deepcopy` Usage in Simulated Annealing

*   **Change:** In SA, calls to `deepcopy(neighbor_solution)` to update `current_solution` were replaced with direct assignments. The `deepcopy` call for `best_solution` was maintained.
*   **Impact:** Reduction of the total script time to **41.42 seconds** (improvement of ~51.7%). SA time decreased to **28.91 seconds** (improvement of ~60.5%).

#### 2.2.3. Vectorization of `is_valid()` and `fitness()` Functions

*   **Changes:** `is_valid()` and `fitness()` functions in `solution.py` were rewritten using NumPy.
*   **Impact:** Total script time with all optimizations was approximately **40.96 seconds**. Vectorization did not bring significant additional improvement in global time in this context but made the code more robust.

### 2.3. Conclusions of the Single-Processor Phase

The single-processor phase was crucial. Optimizing `deepcopy` resulted in a significant reduction in execution time. Vectorization modernized the codebase, preparing it for more demanding scenarios. These results provided valuable information for exploring multiprocessing.

## 3. Multiprocessing Phase - 5 Runs: Initial Evaluation

This section of the project focused on evaluating the performance of optimization algorithms (Hill Climbing, Simulated Annealing, and four configurations of Genetic Algorithms) when executed multiple times in parallel. The main objective of this phase was to obtain more statistically robust data on the effectiveness and efficiency of each approach by running each algorithm 5 times. The use of multiprocessing allowed these concurrent executions, optimizing the total experimentation time.

### 3.1. Experimental Methodology

The evaluated algorithms were: Hill Climbing (HC), Simulated Annealing (SA), and four configurations of Genetic Algorithms (GA_Config_1, GA_Config_2, GA_Config_3, GA_Config_4), varying mutation, crossover, and selection operators. Each algorithm/configuration was executed 5 times in parallel. Collected metrics included best overall fitness, average fitness, standard deviation of fitness, and average execution time.

### 3.2. Results and Discussion (5 Runs)

After executing the `main_script_mp.py` script for 5 runs of each algorithm, the results were:

*   **Best Overall Fitness:** All algorithms (SA, GA_Config_1, GA_Config_2, GA_Config_3, and GA_Config_4) managed to achieve the same Best Overall Fitness value of approximately **0.057143**. Hill Climbing, although it reached this value, had a slightly higher Average Fitness (0.0671).
*   **Consistency (Standard Deviation of Fitness):** Simulated Annealing and GA_Config_4 stood out with a Standard Deviation of Fitness of **0.0**, indicating they converged to the same optimal value in all 5 runs.
*   **Efficiency (Average Execution Time):**
    *   Hill Climbing: ~0.47 seconds (fastest).
    *   GA_Config_1: ~6.05 seconds.
    *   GA_Config_4: ~8.19 seconds.
    *   Simulated Annealing: ~17.60 seconds.
    *   GA_Config_2 and GA_Config_3: ~61-62 seconds (slowest).

### 3.3. Preliminary Conclusions (5 Runs)

Based on the results of 5 runs:

*   **Hill Climbing** proved to be extremely fast.
*   **Simulated Annealing** consistently found the best solution, but at a moderate time cost.
*   **GA_Config_4** emerged as a strong candidate, matching SA in best fitness and consistency, but being faster.
*   **GA_Config_1** was notable for its speed among GAs, albeit with lower consistency.

These results provided the basis for selecting algorithms for the 30-run phase.

## 4. Multiprocessing Phase - 30 Runs with Promising Algorithms: Statistical Analysis

This section of the project presents a detailed statistical analysis of the four algorithms identified as most promising in the previous 5-run phase (Hill Climbing, Simulated Annealing, GA_Config_1, and GA_Config_4). Each algorithm was executed 30 times in parallel to obtain more statistically robust results.

### 4.1. Methodology

The algorithms were executed with their base parameters (the same as in the 5-run phase). Analyzed metrics included average fitness, standard deviation of fitness, best overall fitness, and average execution time.

### 4.2. Results and Statistical Analysis (Initial 30 Runs)

*   **Best Overall Fitness:** All four algorithms (HC, SA, GA_Config_1, GA_Config_4) managed to achieve the same Best Overall Fitness value of approximately **0.0571**.
*   **Average Fitness and Consistency:**
    *   HC, SA, and GA_Config_4 had the same average fitness (0.0605) and the same standard deviation (0.0124).
    *   GA_Config_1 performed slightly worse in average fitness (0.0688) and had higher variability (standard deviation 0.0210).
*   **Efficiency (Average Execution Time):**
    *   Hill Climbing: ~0.63 seconds (fastest).
    *   GA_Config_1: ~5.97 seconds.
    *   GA_Config_4: ~8.30 seconds.
    *   Simulated Annealing: ~18.21 seconds (slowest).

### 4.3. Conclusions (Initial 30 Runs)

*   **Hill Climbing** emerged as the most efficient algorithm, achieving the same solution quality as SA and GA_Config_4, but in a fraction of the time.
*   HC, SA, and GA_Config_4 were equally consistent.
*   HC offered the best balance of quality, consistency, and efficiency.

This analysis led to the decision to focus the final parameter variation phase on Hill Climbing and the two most promising GA configurations (GA_Config_1 and GA_Config_4), discarding SA due to its comparative slowness without a quality gain over HC or the optimized GA_Config_4.

## 5. Parameter Variation Phase - 30 Runs per Variation: Detailed Statistical Analysis

This section details the statistical analysis of the results obtained after executing 30 repetitions for each of the 19 parameter variations applied to Hill Climbing (HC) and the two Genetic Algorithm configurations (GA_Config_1 and GA_Config_4). The objective of this phase was to identify the most effective hyperparameter settings for each algorithm.

### 5.1. Justification for Choosing Genetic Algorithms for Parameter Variation

The `GA_Config_1_SwapConst1PtPreferVTournVarK` and `GA_Config_4_TargetExchUnifPreferVTournVarK_k5` configurations were selected for the detailed parameter variation phase based on their promising performance in the previous round of 30 runs (with base parameters).

*   `GA_Config_1` was the fastest GA configuration in that previous phase that also demonstrated the ability to achieve the best fitness, albeit with lower consistency than others. Its architecture, with `mutate_swap_constrained` and `crossover_one_point_prefer_valid`, represented a more classic and less disruptive approach, whose performance could be fine-tuned.
*   `GA_Config_4` stood out for achieving the best fitness with very good consistency and competitive execution time. Its combination of operators (targeted mutation `mutate_targeted_player_exchange` and uniform crossover `crossover_uniform_prefer_valid`) suggested good potential for exploration and exploitation of the solution space, justifying a more in-depth investigation of its parameters.

The expectation was that by varying parameters such as mutation rate, population size, and number of generations, we could further refine the performance of these two distinct GA architectures.

### 5.2. Parameter Variation Methodology

*   **Hill Climbing (HC):** The **Maximum Number of Iterations** was varied: [500, 1000 (base), 1500].
*   **GA_Config_1 and GA_Config_4:** For each, the following were varied individually (keeping other base parameters at their original values: PopSize=50, NumGen=100, MutRate=0.1):
    *   **Mutation Rate (MutRate):** [0.05, 0.15, 0.25]
    *   **Population Size (PopSize):** [30, 75]
    *   **Number of Generations (NumGen):** [75, 150]
Each of the 19 resulting variations was executed 30 times.

### 5.3. Results and Analysis of Parameter Variation

**Global Key Observation:** All 19 tested configurations, over their 30 runs, were able to find the same best global fitness of **0.0571** at least once.

#### 5.3.1. Optimized Hill Climbing (HC)

*   **HC_Iter_500 (Max Iterations = 500):**
    *   Average Fitness: 0.0588, Standard Deviation: 0.0089, Average Time: 0.41s.
*   **Conclusion for HC:** The `HC_Iter_500` configuration was the most effective, with the best average fitness, lowest standard deviation, and extremely low execution time. Increasing iterations did not bring improvements.

#### 5.3.2. Optimized Genetic Algorithm - Configuration 1 (GA_Config_1)

*   **Best Variations:**
    *   `GA_Config_1_mutation_rate_0.25` (Pop=50, Gen=100): Average Fitness: 0.0588, Standard Deviation: 0.0089, Time: 6.13s.
    *   `GA_Config_1_population_size_75` (MutRate=0.1, Gen=100): Average Fitness: 0.0588, Standard Deviation: 0.0089, Time: 9.08s.
*   **Conclusion for GA_Config_1:** Both variations matched the quality and consistency of `HC_Iter_500`, but with considerably longer execution times. The variation with a mutation rate of 0.25 was more time-efficient.

#### 5.3.3. Optimized Genetic Algorithm - Configuration 4 (GA_Config_4)

*   **Best Variations (consistently achieved average fitness of 0.0571 with standard deviation ~0.0):**
    *   `GA_Config_4_mutation_rate_0.15` (Pop=50, Gen=100): Time: 8.55s.
    *   `GA_Config_4_mutation_rate_0.25` (Pop=50, Gen=100): Time: 8.90s.
    *   `GA_Config_4_population_size_75` (MutRate=0.1, Gen=100): Time: 12.91s.
    *   `GA_Config_4_generations_150` (MutRate=0.1, Pop=50): Time: 12.44s.
*   **Conclusion for GA_Config_4:** Showed remarkable potential, with several variations consistently achieving the best average fitness of 0.0571 with zero standard deviation. `GA_Config_4_mutation_rate_0.15` was the most time-efficient among these top-performing configurations.

### 5.4. General Conclusions from Parameter Variation

1.  **Achievable Best Global Fitness:** 0.0571, reached by all variations.
2.  **Optimized Hill Climbing (`HC_Iter_500`):** Stood out for extreme efficiency (0.41s), excellent average fitness (0.0588), and good consistency.
3.  **Optimized GA_Config_1 (`GA_Config_1_mutation_rate_0.25`):** Matched the quality of optimized HC, but slower (6.13s).
4.  **Optimized GA_Config_4 (`GA_Config_4_mutation_rate_0.15`):** Consistently achieved the best average fitness (0.0571) with zero standard deviation, being the fastest (8.55s) among those that reached this level of perfection.

**Final Algorithm and Configuration Recommendations Post-Variation:**

*   **For Maximum Efficiency with Excellent Quality:** `HC_Iter_500`.
*   **To Guarantee Best Fitness with Perfect Consistency:** `GA_Config_4_mutation_rate_0.15`.

## 6. Final Conclusions and Recommendations

### 6.1. Synthesis of Results

This project explored the application of metaheuristic algorithms to the optimization problem of sports team selection, progressing from single-processor implementations to detailed parameter variation analyses. The main results can be summarized as follows:

1. **Code Optimizations:** Reducing `deepcopy` usage in Simulated Annealing resulted in a performance improvement of approximately 51.7% in total execution time.

2. **Multiprocessing - 5 Runs:** All algorithms managed to achieve the same best overall fitness (0.057143), but with significant differences in consistency and execution time. Hill Climbing was the fastest (0.47s), while Simulated Annealing and GA_Config_4 were the most consistent.

3. **Multiprocessing - 30 Runs:** Hill Climbing, Simulated Annealing, and GA_Config_4 showed statistically equivalent performance in terms of solution quality (average fitness 0.0605, standard deviation 0.0124), but Hill Climbing was significantly faster (0.63s vs. 18.21s for SA).

4. **Parameter Variation - 30 Runs per Variation:**
   - Hill Climbing with 500 iterations (`HC_Iter_500`) emerged as the most efficient configuration, with excellent average fitness (0.0588), good consistency (standard deviation 0.0089), and extremely low time (0.41s).
   - GA_Config_4 with a mutation rate of 0.15 (`GA_Config_4_mutation_rate_0.15`) consistently achieved the best average fitness (0.0571) with zero standard deviation, being the fastest (8.55s) among configurations guaranteeing this level of perfection.

### 6.2. Recommendations for Practical Application

Based on the results obtained, we recommend:

1. **For Time-Constrained Scenarios:** Use Hill Climbing with 500 iterations (`HC_Iter_500`), which offers an excellent balance between solution quality and computational efficiency.

2. **For Scenarios Requiring Guaranteed Best Solution:** Use Genetic Algorithm GA_Config_4 with a mutation rate of 0.15 (`GA_Config_4_mutation_rate_0.15`), which consistently guarantees the best possible fitness, albeit at a higher computational cost.

3. **For Large-Scale Applications:** Consider implementing a hybrid approach, starting with Hill Climbing to quickly obtain a good solution, followed by refinement with optimized GA_Config_4 for specific cases where maximum quality is critical.

### 6.3. Suggestions for Future Work

1. **Exploration of Algorithm Hybridization:** Investigate combining Hill Climbing with Genetic Algorithms, potentially using HC to refine solutions generated by GAs.

2. **Scalability Analysis:** Test the performance of optimized algorithms on larger-scale problems, with more teams or more players.

3. **Internal Algorithm Parallelization:** In addition to multiprocessing for multiple runs, explore internal algorithm parallelization, especially for Genetic Algorithms (parallel evaluation of individuals).

4. **Exploration of Machine Learning Techniques:** Investigate the use of learning techniques to predict good parameter configurations based on problem characteristics.

## 7. Analysis Process Diagram

The diagram below illustrates the analysis process followed in this project, from initial implementation to final recommendations:

```
┌─────────────────────────┐
│ Initial Implementation  │
│ (Single-Processor)      │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Profiling & Optimization│
│ - Reduce deepcopy       │
│ - Vectorization         │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Multiprocessing         │
│ (5 Runs)                │
│ - HC, SA, 4 GA configs  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Selection of Promising  │
│ Algorithms              │
│ - HC, SA, GA1, GA4      │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Statistical Analysis    │
│ (30 Runs)               │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Selection for Parameter │
│ Variation               │
│ - HC, GA1, GA4          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Parameter Variation     │
│ (30 Runs/Variation)     │
│ - 19 configurations     │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│ Final Analysis &        │
│ Recommendations         │
│ - HC_Iter_500           │
│ - GA4_MutRate_0.15      │
└─────────────────────────┘
```

This methodological process allowed for a systematic exploration of the algorithm and parameter space, culminating in well-founded recommendations for practical application.
