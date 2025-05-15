# Statistical Analysis of Parameter Variation Results (30 Runs)

This section details the statistical analysis of the results obtained after executing 30 repetitions for each of the 19 parameter variations of the Hill Climbing (HC), GA_Config_1, and GA_Config_4 algorithms. The objective is to identify the parameter configurations that optimize the performance of each algorithm and subsequently compare the best-optimized algorithms.

The complete data from this experimentation phase has been saved in `/home/ubuntu/CIFO_EXTENDED_Project/images_mp/final_param_var_results/all_algorithms_summary_final_param_var_30runs.csv`.

## Analysis Methodology

The analysis focused on three primary metrics for each algorithm variation:

1.  **Overall Best Fitness:** The best solution (lowest standard deviation) found across all 30 executions.
2.  **Mean Fitness:** The average of the best fitness values found in each of the 30 executions. This indicates the typical performance of the algorithm.
3.  **Standard Deviation of Fitness (Std Dev Fitness):** A measure of the algorithm's consistency. Lower values indicate that the algorithm consistently finds solutions of similar quality.
4.  **Mean Execution Time:** The average time each algorithm execution took to complete.

Given the nature of the data and the goal of identifying the highest-performing configurations, the comparative analysis will be primarily qualitative, observing trends and extreme values in the metrics above. For a formal analysis, normality tests (e.g., Shapiro-Wilk) would be applied, followed by parametric (ANOVA, t-test) or non-parametric (Kruskal-Wallis, Mann-Whitney U) tests with post-hoc corrections, depending on the data distribution. However, for this report, we will focus on the direct interpretation of the summarized results.

## Results and Analysis by Algorithm

### 1. Hill Climbing (HC)

Three variations for the maximum number of iterations were tested:

*   **HC_MaxIter_500:**
    *   Mean Fitness: 0.0571
    *   Std Dev Fitness: 1.38e-17 (essentially zero, indicating perfect consistency)
    *   Overall Best Fitness: 0.0571
    *   Mean Time: 0.44s
*   **HC_MaxIter_1000:**
    *   Mean Fitness: 0.0654
    *   Std Dev Fitness: 0.0185
    *   Overall Best Fitness: 0.0571
    *   Mean Time: 0.41s
*   **HC_MaxIter_1500:**
    *   Mean Fitness: 0.0605
    *   Std Dev Fitness: 0.0124
    *   Overall Best Fitness: 0.0571
    *   Mean Time: 0.43s

**Conclusion for HC:** The `HC_MaxIter_500` configuration proved to be ideal. It achieved the best possible fitness (0.0571) with perfect consistency (practically null standard deviation) and very low execution time. Increasing the number of iterations to 1000 or 1500 did not improve the quality of the best solution found and, in some cases (like MaxIter_1000), even worsened the mean fitness and consistency, suggesting that HC converges very quickly to the optimum (or a high-quality local optimum) and additional iterations bring no benefit.

### 2. Genetic Algorithm - Configuration 1 (GA_Config_1_SwapConst1PtPreferVTournVarK)

The base GA_Config_1 (Pop=50, Gen=100, MutRate=0.1) showed:
*   Mean Fitness: 0.0688
*   Std Dev Fitness: 0.0210
*   Overall Best Fitness: 0.0571
*   Mean Time: 5.98s

Analyzing the variations:

*   **Mutation Rate Variation (MutRate):**
    *   `MutRate=0.05`: Mean Fitness 0.0781 (worse), StdDev 0.0264, Time 5.84s.
    *   `MutRate=0.15`: Mean Fitness 0.0638 (better than base), StdDev 0.0169, Time 6.11s.
    *   `MutRate=0.25`: Mean Fitness 0.0571 (excellent), StdDev 1.38e-17 (perfect consistency), Time 6.25s.
    A mutation rate of 0.25 resulted in notably superior performance, achieving the best fitness with perfect consistency.

*   **Population Size Variation (PopSize):**
    *   `PopSize=30`: Mean Fitness 0.0699 (slightly worse than base), StdDev 0.0237, Time 3.51s (faster).
    *   `PopSize=75`: Mean Fitness 0.0571 (excellent), StdDev 1.38e-17 (perfect consistency), Time 9.04s (slower).
    Increasing the population size to 75 drastically improved quality and consistency, at the cost of longer execution time.

*   **Number of Generations Variation (NumGen):**
    *   `NumGen=75`: Mean Fitness 0.0654 (slightly better than base), StdDev 0.0185, Time 4.56s (faster).
    *   `NumGen=150`: Mean Fitness 0.0638 (better than base), StdDev 0.0169, Time 8.90s (slower).
    Increasing generations to 150 improved mean fitness and consistency compared to the base.

**Best Configurations for GA_Config_1:**
The variations `GA_Config_1_mutation_rate_0.25` (Pop=50, Gen=100, MutRate=0.25) and `GA_Config_1_population_size_75` (Pop=75, Gen=100, MutRate=0.1) stood out, both achieving a mean fitness of 0.0571 with perfect consistency. The former is faster (6.25s vs 9.04s).

### 3. Genetic Algorithm - Configuration 4 (GA_Config_4_TargetExchUnifPreferVTournVarK_k5)

The base GA_Config_4 (Pop=50, Gen=100, MutRate=0.1) showed:
*   Mean Fitness: 0.0605
*   Std Dev Fitness: 0.0124
*   Overall Best Fitness: 0.0571
*   Mean Time: 8.21s

Analyzing the variations:

*   **Mutation Rate Variation (MutRate):**
    *   `MutRate=0.05`: Mean Fitness 0.0605 (same as base), StdDev 0.0124, Time 8.30s.
    *   `MutRate=0.15`: Mean Fitness 0.0571 (excellent), StdDev 1.38e-17 (perfect consistency), Time 8.67s.
    *   `MutRate=0.25`: Mean Fitness 0.0571 (excellent), StdDev 1.38e-17 (perfect consistency), Time 8.88s.
    Mutation rates of 0.15 and 0.25 were optimal, with 0.15 being slightly faster.

*   **Population Size Variation (PopSize):**
    *   `PopSize=30`: Mean Fitness 0.0691 (worse than base), StdDev 0.0259, Time 4.83s (faster).
    *   `PopSize=75`: Mean Fitness 0.0571 (excellent), StdDev 1.38e-17 (perfect consistency), Time 12.74s (slower).
    Increasing the population to 75 was beneficial for quality and consistency but increased time.

*   **Number of Generations Variation (NumGen):**
    *   `NumGen=75`: Mean Fitness 0.0654 (worse than base), StdDev 0.0185, Time 6.41s (faster).
    *   `NumGen=150`: Mean Fitness 0.0588 (better than base), StdDev 0.0089 (good consistency), Time 12.47s (slower).
    Increasing generations to 150 improved mean fitness and consistency.

**Best Configurations for GA_Config_4:**
The variations `GA_Config_4_mutation_rate_0.15` (Pop=50, Gen=100, MutRate=0.15), `GA_Config_4_mutation_rate_0.25` (Pop=50, Gen=100, MutRate=0.25), and `GA_Config_4_population_size_75` (Pop=75, Gen=100, MutRate=0.1) were the best, all achieving a mean fitness of 0.0571 with perfect consistency. Among these, `GA_Config_4_mutation_rate_0.15` was the fastest (8.67s).

## General Conclusions from Parameter Variation

1.  **Hill Climbing:** The `HC_MaxIter_500` configuration is extremely efficient and robust, consistently finding the best-known solution (0.0571) in under 0.5 seconds.

2.  **Optimized GA_Config_1:** The `GA_Config_1_mutation_rate_0.25` configuration (Pop=50, Gen=100, MutRate=0.25) proved to be the best for this GA architecture, achieving the best fitness (0.0571) with perfect consistency and a mean time of 6.25s.

3.  **Optimized GA_Config_4:** The `GA_Config_4_mutation_rate_0.15` configuration (Pop=50, Gen=100, MutRate=0.15) was the most balanced for this architecture, also achieving the best fitness (0.0571) with perfect consistency, in a mean time of 8.67s.

Comparing the best-optimized performances:

*   **HC_MaxIter_500:** Mean Fitness 0.0571, StdDev ~0, Time ~0.44s.
*   **GA_Config_1_mutation_rate_0.25:** Mean Fitness 0.0571, StdDev ~0, Time ~6.25s.
*   **GA_Config_4_mutation_rate_0.15:** Mean Fitness 0.0571, StdDev ~0, Time ~8.67s.

All three algorithms, with their optimized parameters, can consistently find the best-known solution. Hill Climbing remains orders of magnitude faster. Among the GAs, the optimized GA_Config_1 is slightly faster than the optimized GA_Config_4, both offering excellent solution quality and consistency.

The final choice of algorithm will depend on the desired balance between execution time and the confidence of exploring different parts of the solution space that GAs can offer compared to the more local nature of HC. However, for this specific problem and with the results obtained, **HC_MaxIter_500** presents itself as the most pragmatic and efficient solution.

