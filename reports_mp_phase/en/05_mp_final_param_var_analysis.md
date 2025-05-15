# Statistical Analysis of Parameter Variation Results (30 Runs per Variation)

This section details the statistical analysis of the results obtained after executing 30 repetitions for each of the 19 parameter variations applied to Hill Climbing (HC) and the two Genetic Algorithm (GA) configurations (GA_Config_1 and GA_Config_4). The objective of this phase was to identify the most effective hyperparameter configurations for each algorithm, focusing on solution quality (mean fitness), consistency (standard deviation of fitness), and computational efficiency (mean execution time).

All results and convergence plots have been saved in the directory `/home/ubuntu/CIFO_EXTENDED_Project/images_mp/final_param_var_results/`, and the consolidated summary can be found in `all_algorithms_summary_final_param_var_30runs.csv`.

## Global Key Observation

A notable result from this extensive parameter variation phase is that **all 19 tested configurations, across their 30 runs, were able to find the same best overall fitness of 0.0571 at least once**. This suggests that this value may represent a global optimum or, at least, a very strong and consistently achievable local optimum by the tested algorithms and their variations.

## Detailed Analysis by Algorithm

### 1. Hill Climbing (HC)

For Hill Climbing, the maximum number of iterations was varied. The tested configurations and their average performances over 30 runs were:

*   **HC_Iter_500 (Max Iterations = 500):**
    *   Mean Fitness: 0.0588
    *   Standard Deviation of Fitness: 0.0089
    *   Mean Execution Time: 0.41 seconds
*   **HC_Iter_1000 (Max Iterations = 1000 - Base):**
    *   Mean Fitness: 0.0605
    *   Standard Deviation of Fitness: 0.0124
    *   Mean Execution Time: 0.42 seconds
*   **HC_Iter_1500 (Max Iterations = 1500):**
    *   Mean Fitness: 0.0638
    *   Standard Deviation of Fitness: 0.0169
    *   Mean Execution Time: 0.41 seconds

**Conclusion for Hill Climbing:**
The `HC_Iter_500` configuration proved to be the most effective among the tested variations. It achieved the best mean fitness (0.0588) and the lowest standard deviation (0.0089), indicating good solution quality and consistency, all while maintaining an extremely low execution time (0.41s). Increasing the number of iterations to 1000 or 1500 did not result in an improvement in mean fitness or the best overall fitness found, and in fact, led to a slight deterioration in mean fitness and increased variability. Therefore, 500 iterations seem sufficient for HC to effectively explore the search space for this problem.

### 2. Genetic Algorithm - Configuration 1 (GA_Config_1)

This configuration uses the `mutate_swap_constrained` mutation operator, `crossover_one_point_prefer_valid` crossover, and tournament selection with k=3. The base parameters were: Population Size = 50, Number of Generations = 100, Mutation Rate = 0.1.

Results of variations (mean fitness, standard deviation, mean time):

*   **GA_Config_1_Base:** (Pop=50, Gen=100, MutRate=0.1)
    *   Mean Fitness: 0.0638, Standard Deviation: 0.0169, Time: 5.91s
*   **Mutation Rate Variations (MutRate):**
    *   `MutRate=0.05`: Mean Fitness: 0.0688, Standard Deviation: 0.0210, Time: 5.78s (Worsened)
    *   `MutRate=0.15`: Mean Fitness: 0.0605, Standard Deviation: 0.0124, Time: 5.99s (Improved)
    *   `MutRate=0.25`: Mean Fitness: 0.0588, Standard Deviation: 0.0089, Time: 6.13s (**Best MutRate variation for GA1**)
*   **Population Size Variations (PopSize):**
    *   `PopSize=30`: Mean Fitness: 0.0732, Standard Deviation: 0.0251, Time: 3.49s (Worsened, but faster)
    *   `PopSize=75`: Mean Fitness: 0.0588, Standard Deviation: 0.0089, Time: 9.08s (**Matched best MutRate in quality/consistency, but slower**)
*   **Number of Generations Variations (NumGen):**
    *   `NumGen=75`: Mean Fitness: 0.0682, Standard Deviation: 0.0228, Time: 4.47s (Worsened, but faster)
    *   `NumGen=150`: Mean Fitness: 0.0605, Standard Deviation: 0.0124, Time: 9.10s (Improved, but slower)

**Conclusion for GA_Config_1:**
The most promising variations for `GA_Config_1` were `GA_Config_1_mutation_rate_0.25` (Pop=50, Gen=100) and `GA_Config_1_population_size_75` (MutRate=0.1, Gen=100). Both achieved a mean fitness of 0.0588 and a standard deviation of 0.0089, matching the quality and consistency of the best Hill Climbing configuration. However, their execution times were considerably higher (6.13s and 9.08s, respectively). Between these two, the variation with a mutation rate of 0.25 was more time-efficient.

### 3. Genetic Algorithm - Configuration 4 (GA_Config_4)

This configuration uses the `mutate_targeted_player_exchange` mutation operator, `crossover_uniform_prefer_valid` crossover, and tournament selection with k=5. The base parameters were: Population Size = 50, Number of Generations = 100, Mutation Rate = 0.1.

Results of variations (mean fitness, standard deviation, mean time):

*   **GA_Config_4_Base:** (Pop=50, Gen=100, MutRate=0.1)
    *   Mean Fitness: 0.0621, Standard Deviation: 0.0149, Time: 8.45s
*   **Mutation Rate Variations (MutRate):**
    *   `MutRate=0.05`: Mean Fitness: 0.0621, Standard Deviation: 0.0149, Time: 9.50s (Similar to base, slower)
    *   `MutRate=0.15`: Mean Fitness: 0.0571, Standard Deviation: ~0.0000, Time: 8.55s (**Excellent: perfect fitness, perfect consistency**)
    *   `MutRate=0.25`: Mean Fitness: 0.0571, Standard Deviation: ~0.0000, Time: 8.90s (**Excellent: perfect fitness, perfect consistency**)
*   **Population Size Variations (PopSize):**
    *   `PopSize=30`: Mean Fitness: 0.0638, Standard Deviation: 0.0169, Time: 4.80s (Worsened, but faster)
    *   `PopSize=75`: Mean Fitness: 0.0571, Standard Deviation: ~0.0000, Time: 12.91s (**Excellent: perfect fitness, perfect consistency, but slower**)
*   **Number of Generations Variations (NumGen):**
    *   `NumGen=75`: Mean Fitness: 0.0638, Standard Deviation: 0.0169, Time: 6.40s (Worsened, but faster)
    *   `NumGen=150`: Mean Fitness: 0.0571, Standard Deviation: ~0.0000, Time: 12.44s (**Excellent: perfect fitness, perfect consistency, but slower**)

**Conclusion for GA_Config_4:**
`GA_Config_4` demonstrated remarkable potential with parameter tuning. Several of its variations (`MutRate=0.15`, `MutRate=0.25`, `PopSize=75`, `NumGen=150`) consistently achieved the best mean fitness of 0.0571 (equal to the best overall fitness found) with a practically null standard deviation. This indicates a very robust convergence to the high-quality solution.
Among these top configurations, `GA_Config_4_mutation_rate_0.15` (Pop=50, Gen=100, MutRate=0.15) was the most time-efficient, achieving this performance in an average of 8.55 seconds.

## General Conclusions from Parameter Variation

1.  **Achievable Best Overall Fitness:** All algorithms and their variations were capable of finding the best fitness of 0.0571, suggesting this is a robust benchmark for the problem.

2.  **Optimized Hill Climbing:** The `HC_Iter_500` configuration (500 iterations) stood out for its extreme efficiency (0.41s), achieving excellent mean fitness (0.0588) and good consistency (standard deviation 0.0089).

3.  **Optimized GA_Config_1:** The best variation was `GA_Config_1_mutation_rate_0.25`, which matched the quality and consistency of `HC_Iter_500` (mean fitness 0.0588, standard deviation 0.0089), but at a significantly higher computational cost (6.13s).

4.  **Optimized GA_Config_4:** This configuration, especially with `MutRate=0.15` (or 0.25), `PopSize=75`, or `NumGen=150`, demonstrated the ability to consistently achieve the best mean fitness of 0.0571 with null standard deviation. The `GA_Config_4_mutation_rate_0.15` variant was the fastest among these (8.55s) to reach this level of perfection.

**Final Recommendation of Algorithms and Configurations:**

*   **For Maximum Efficiency with Excellent Quality:** `HC_Iter_500` is the clear choice. It offers a high-quality solution very quickly and consistently.
*   **For Guaranteeing Best Fitness with Perfect Consistency:** If the goal is to achieve a fitness of 0.0571 with maximum consistency, `GA_Config_4_mutation_rate_0.15` (Pop=50, Gen=100, MutRate=0.15) is the best option, although it is considerably slower than HC.

The choice between these will depend on the specific problem requirements in terms of execution time versus the need to guarantee the best possible score in all runs.

## Justification for Selecting Genetic Algorithms for Parameter Variation

The `GA_Config_1` and `GA_Config_4` configurations were selected for the detailed parameter variation phase based on their promising performance in the previous round of 30 runs (with base parameters).

*   `GA_Config_1_SwapConst1PtPreferVTournVarK` was the fastest GA configuration in that earlier phase that also demonstrated the ability to achieve the best fitness, albeit with less consistency than others.
*   `GA_Config_4_TargetExchUnifPreferVTournVarK_k5` stood out for achieving the best fitness with very good consistency (low standard deviation) and an execution time that, while higher than HC, was competitive against other slower and less consistent GA configurations. Its combination of operators (targeted mutation and uniform crossover with preference for valid individuals) suggested good potential for exploring and exploiting the solution space.

The expectation was that by varying parameters such as mutation rate, population size, and number of generations, we could further refine the performance of these two distinct GA architectures, exploring different balances between exploration and exploitation.

