# Analysis of Parameter Variation Results for Genetic Algorithms

This report details the analysis of results from parameter variation experiments conducted on two promising Genetic Algorithm (GA) configurations: GA_Config_1 and GA_Config_4. The objective was to identify hyperparameter settings that optimize the performance of these algorithms in terms of solution quality (mean fitness), consistency (standard deviation of fitness), and efficiency (mean execution time).

The tested variations focused on Mutation Rate (MutRate), Population Size (PopSize), and Number of Generations (NumGen), with each variation executed 5 times.

## Analysis Methodology

Data was collected from the `all_ga_variations_summary_mp_5runs.csv` file. For each base configuration (GA_Config_1 and GA_Config_4), the impact of each varied parameter was analyzed while keeping other parameters at their base values (PopSize=50, NumGen=100, MutRate=0.1, unless the parameter itself was being varied).

## Results and Analysis for GA_Config_1

The base GA_Config_1 uses `mutate_swap_constrained`, `crossover_one_point_prefer_valid`, and `selection_tournament_variable_k` (k=3).

### Impact of Mutation Rate (MutRate) on GA_Config_1 (PopSize=50, NumGen=100)

| MutRate | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|---------|--------------|-----------------|--------------------|
| 0.05    | 0.0770       | 0.0244          | 5.68               |
| **0.10 (Base)** | **0.0688**   | **0.0210**      | **5.97**           | (Result from previous 30 runs for reference)
| 0.15    | 0.0571       | 0.0000          | 6.07               |
| 0.25    | 0.0671       | 0.0199          | 6.31               |

**Observations for GA_Config_1 - MutRate:**
A mutation rate of 0.15 resulted in the best mean fitness (0.0571) and perfect consistency (standard deviation of 0.0), outperforming the base rate of 0.1. Lower (0.05) or higher (0.25) rates led to slightly worse mean fitness and lower consistency. Execution time increased marginally with higher mutation rates.

### Impact of Population Size (PopSize) on GA_Config_1 (MutRate=0.1, NumGen=100)

| PopSize | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|---------|--------------|-----------------|--------------------|
| 30      | 0.0870       | 0.0244          | 3.48               |
| **50 (Base)** | **0.0688**   | **0.0210**      | **5.97**           |
| 75      | 0.0770       | 0.0244          | 9.02               |

**Observations for GA_Config_1 - PopSize:**
The base population of 50 individuals provided the best compromise. Reducing the population to 30 significantly worsened the mean fitness, although it was faster. Increasing to 75 did not improve mean fitness compared to the base and considerably increased execution time.

### Impact of Number of Generations (NumGen) on GA_Config_1 (MutRate=0.1, PopSize=50)

| NumGen | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|--------|--------------|-----------------|--------------------|
| 75     | 0.0571       | 0.0000          | 4.43               |
| **100 (Base)**| **0.0688**   | **0.0210**      | **5.97**           |
| 150    | 0.0770       | 0.0244          | 8.84               |

**Observations for GA_Config_1 - NumGen:**
Reducing the number of generations to 75 resulted in the best mean fitness (0.0571) and perfect consistency, in addition to being faster than the base configuration. Increasing the number of generations to 150 brought no improvements in mean fitness and increased execution time.

### Best Variation for GA_Config_1:
Considering individual variations, `GA_Config_1_NumGen_75` (MutRate=0.1, PopSize=50, NumGen=75) and `GA_Config_1_MutRate_0.15` (MutRate=0.15, PopSize=50, NumGen=100) stood out, both achieving a mean fitness of 0.0571 with a standard deviation of 0.0. `GA_Config_1_NumGen_75` was faster (4.43s vs 6.07s).

## Results and Analysis for GA_Config_4

The base GA_Config_4 uses `mutate_targeted_player_exchange`, `crossover_uniform_prefer_valid`, and `selection_tournament_variable_k` (k=5).

### Impact of Mutation Rate (MutRate) on GA_Config_4 (PopSize=50, NumGen=100)

| MutRate | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|---------|--------------|-----------------|--------------------|
| 0.05    | 0.0671       | 0.0199          | 8.12               |
| **0.10 (Base)** | **0.0605**   | **0.0124**      | **8.30**           | (Result from previous 30 runs for reference)
| 0.15    | 0.0571       | 0.0000          | 8.89               |
| 0.25    | 0.0571       | 0.0000          | 8.68               |

**Observations for GA_Config_4 - MutRate:**
Mutation rates of 0.15 and 0.25 resulted in the best mean fitness (0.0571) and perfect consistency, outperforming the base rate of 0.1. Execution time increased slightly with these higher rates.

### Impact of Population Size (PopSize) on GA_Config_4 (MutRate=0.1, NumGen=100)

| PopSize | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|---------|--------------|-----------------|--------------------|
| 30      | 0.0571       | 0.0000          | 4.96               |
| **50 (Base)** | **0.0605**   | **0.0124**      | **8.30**           |
| 75      | 0.0571       | 0.0000          | 12.93              |

**Observations for GA_Config_4 - PopSize:**
Reducing the population size to 30 improved the mean fitness to 0.0571 with perfect consistency and significantly reduced execution time (4.96s). Increasing to 75 also achieved optimal fitness and perfect consistency, but at a much higher execution time cost.

### Impact of Number of Generations (NumGen) on GA_Config_4 (MutRate=0.1, PopSize=50)

| NumGen | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|--------|--------------|-----------------|--------------------|
| 75     | 0.0571       | 0.0000          | 6.28               |
| **100 (Base)**| **0.0605**   | **0.0124**      | **8.30**           |
| 150    | 0.0571       | 0.0000          | 12.32              |

**Observations for GA_Config_4 - NumGen:**
Reducing the number of generations to 75 improved the mean fitness to 0.0571 with perfect consistency and reduced execution time (6.28s). Increasing to 150 also achieved optimal fitness and consistency, but with a longer execution time.

### Best Variation for GA_Config_4:
The `GA_Config_4_PopSize_30` variation (MutRate=0.1, PopSize=30, NumGen=100) was the most efficient, achieving optimal fitness of 0.0571 with perfect consistency in the shortest execution time (4.96s). Other variations like `GA_Config_4_NumGen_75` and those with `MutRate` of 0.15 or 0.25 were also excellent in terms of fitness and consistency but slower.

## Conclusions from Parameter Variation

1.  **Achievable Consistency:** Many variations, especially for GA_Config_4, managed to achieve the best-known fitness (0.0571) with perfect consistency (standard deviation 0.0) in the 5 runs, which is a very positive result.

2.  **Optimized GA_Config_1:** The `GA_Config_1_NumGen_75` configuration (MutRate=0.1, PopSize=50, NumGen=75) showed a significant improvement over the base, achieving optimal fitness and perfect consistency with an execution time of 4.43s.

3.  **Optimized GA_Config_4:** The `GA_Config_4_PopSize_30` configuration (MutRate=0.1, PopSize=30, NumGen=100) stood out as the most efficient for GA_Config_4, with optimal fitness, perfect consistency, and an execution time of 4.96s.

4.  **Impact of Parameters:**
    *   **Mutation Rate:** For both configurations, a mutation rate slightly higher than the base (0.15) seemed beneficial for consistently achieving the best fitness.
    *   **Population Size:** For GA_Config_4, reducing the population to 30 was advantageous. For GA_Config_1, the base of 50 was still better than 30 or 75.
    *   **Number of Generations:** Reducing the number of generations to 75 was beneficial for both configurations, suggesting that convergence to the best solution can occur earlier than the 100 base generations, especially with other well-tuned parameters.

5.  **Comparison between Optimized GA_Config_1 and GA_Config_4:**
    *   `GA_Config_1_NumGen_75`: Mean Fitness 0.0571, StdDev 0.0, Time 4.43s
    *   `GA_Config_4_PopSize_30`: Mean Fitness 0.0571, StdDev 0.0, Time 4.96s

    Both optimized configurations achieved identical results in terms of solution quality and consistency. `GA_Config_1_NumGen_75` was marginally faster.

## Recommendations for Future Runs

Based on this parameter variation analysis, the following GA configurations are recommended for future consideration if GAs are chosen over Hill Climbing (which proved more efficient in the previous phase):

*   **Option 1 (Based on GA_Config_1):**
    *   Operators: `mutate_swap_constrained`, `crossover_one_point_prefer_valid`, `selection_tournament_variable_k` (k=3)
    *   Parameters: PopSize=50, NumGen=75, MutRate=0.1 (or 0.15 for potentially greater robustness, with a slight increase in time)

*   **Option 2 (Based on GA_Config_4):**
    *   Operators: `mutate_targeted_player_exchange`, `crossover_uniform_prefer_valid`, `selection_tournament_variable_k` (k=5)
    *   Parameters: PopSize=30, NumGen=100 (or 75), MutRate=0.1 (or 0.15 / 0.25 for consistency, with an increase in time)

It is important to note that Hill Climbing, in the previous analysis of 30 runs, consistently achieved a fitness of 0.0571 in about 0.63 seconds. The GA configurations optimized here, while consistent, are still considerably slower (4.43s - 4.96s). The final choice will depend on the desired balance between the exploration guarantee of GAs and the efficiency of HC for this specific problem.

