# Statistical Analysis of Promising Algorithms (30 Runs)

## 1. Introduction

This section of the project presents a detailed statistical analysis of the four algorithms identified as most promising in the previous 5-run phase. Each algorithm was executed 30 times in parallel, using multiprocessing, to obtain statistically more robust and reliable results. This analysis is crucial for determining not only which algorithm achieves the best solution but also which presents the best balance between solution quality, consistency, and computational efficiency.

## 2. Methodology

### 2.1. Evaluated Algorithms

Based on the results of the previous phase (5 runs), the following algorithms were selected for this more in-depth analysis:

1.  **Hill Climbing (HC)**: Selected for its extreme speed, despite lower consistency in mean fitness during the initial 5 runs.
2.  **Simulated Annealing (SA)**: Chosen for its ability to consistently find the best fitness, with a moderate execution time.
3.  **GA_Config_1_SwapConst1PtPreferVTournVarK**: Selected for being the fastest Genetic Algorithm configuration that also achieved the best fitness.
4.  **GA_Config_4_TargetExchUnifPreferVTournVarK_k5**: Chosen for achieving the best fitness with perfect consistency and a reasonable execution time, being faster than SA.

### 2.2. Execution Parameters

Each algorithm was executed 30 times with the following fixed parameters:

-   **Hill Climbing**:
    -   Maximum iterations: 1000

-   **Simulated Annealing**:
    -   Initial temperature: 1000
    -   Final temperature: 0.1
    -   Cooling rate (alpha): 0.99
    -   Iterations per temperature: 50

-   **GA_Config_1_SwapConst1PtPreferVTournVarK**:
    -   Population size: 50
    -   Number of generations: 100
    -   Mutation operator: `mutate_swap_constrained`
    -   Mutation rate: 0.1
    -   Crossover operator: `crossover_one_point_prefer_valid`
    -   Selection operator: `selection_tournament_variable_k` (with k=3)
    -   Elitism size: 2

-   **GA_Config_4_TargetExchUnifPreferVTournVarK_k5**:
    -   Population size: 50
    -   Number of generations: 100
    -   Mutation operator: `mutate_targeted_player_exchange`
    -   Mutation rate: 0.1
    -   Crossover operator: `crossover_uniform_prefer_valid`
    -   Selection operator: `selection_tournament_variable_k` (with k=5)
    -   Elitism size: 3

### 2.3. Analyzed Metrics

For each algorithm, the following metrics were analyzed:

-   **Mean Fitness**: The average of the best fitness values obtained in the 30 runs.
-   **Standard Deviation of Fitness**: A measure of the algorithm's consistency in achieving good results.
-   **Overall Best Fitness**: The lowest value of the fitness function achieved in all runs.
-   **Mean Execution Time**: The average time (in seconds) each algorithm took to complete one run.

## 3. Results and Statistical Analysis

### 3.1. Summary of Results

The following table summarizes the results obtained for the four algorithms over the 30 runs:

| Algorithm             | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) | Overall Best Fitness |
|-----------------------|--------------|-----------------|--------------------|----------------------|
| Hill Climbing         | 0.0605       | 0.0124          | 0.63               | 0.0571               |
| Simulated Annealing   | 0.0605       | 0.0124          | 18.21              | 0.0571               |
| GA_Config_1           | 0.0688       | 0.0210          | 5.97               | 0.0571               |
| GA_Config_4           | 0.0605       | 0.0124          | 8.30               | 0.0571               |

### 3.2. Solution Quality Analysis (Fitness)

A notable result is that all algorithms managed to achieve the same **Overall Best Fitness** value of approximately **0.057143**. This suggests that this value may represent the global optimum for the problem with the given data, or at least a very robust local optimum that all algorithms can find.

In terms of **Mean Fitness**, an interesting pattern is observed: Hill Climbing, Simulated Annealing, and GA_Config_4 show exactly the same value (0.0605), while GA_Config_1 performs slightly worse (0.0688). This indicates that, on average, three of the four algorithms have the same ability to consistently find good solutions.

The **Standard Deviation of Fitness** follows the same pattern: Hill Climbing, Simulated Annealing, and GA_Config_4 have the same value (0.0124), while GA_Config_1 shows greater variability (0.0210). This suggests that the first three algorithms have the same consistency in the quality of the solutions found, while GA_Config_1 is less consistent.

### 3.3. Efficiency Analysis (Execution Time)

Regarding **Mean Execution Time**, the differences are significant:

-   **Hill Climbing** is by far the fastest, with an average time of only **0.63 seconds** per run.
-   **GA_Config_1** is the second fastest, with an average time of **5.97 seconds**.
-   **GA_Config_4** ranks third, with an average time of **8.30 seconds**.
-   **Simulated Annealing** is the slowest, with an average time of **18.21 seconds**.

These results show a clear trade-off among the algorithms: Hill Climbing is extremely fast but not necessarily more accurate than the others; Simulated Annealing is very slow but consistent; and the GA configurations offer different balances between speed and consistency.

### 3.4. Detailed Comparative Analysis

#### 3.4.1. Hill Climbing vs. Simulated Annealing

Hill Climbing and Simulated Annealing show exactly the same mean fitness and standard deviation values, which is an interesting result. The main difference lies in execution time: Hill Climbing is approximately 29 times faster than Simulated Annealing. This suggests that, for this specific problem, Hill Climbing may be preferable to Simulated Annealing, as it achieves the same solution quality in much less time.

#### 3.4.2. GA_Config_1 vs. GA_Config_4

Between the two Genetic Algorithm configurations, GA_Config_4 clearly outperforms GA_Config_1 in terms of solution quality (lower mean fitness) and consistency (lower standard deviation). However, GA_Config_1 is approximately 1.4 times faster. The choice between these two configurations will depend on the relative importance of solution quality versus execution time.

#### 3.4.3. Hill Climbing vs. GA_Config_4

Hill Climbing and GA_Config_4 have exactly the same mean fitness and standard deviation values, but Hill Climbing is approximately 13 times faster. This suggests that, for this specific problem, Hill Climbing may be preferable to GA_Config_4, unless there are specific reasons to prefer a population-based approach.

#### 3.4.4. Simulated Annealing vs. GA_Config_4

Simulated Annealing and GA_Config_4 also have the same mean fitness and standard deviation values, but GA_Config_4 is approximately 2.2 times faster. This suggests that, for this specific problem, GA_Config_4 may be preferable to Simulated Annealing.

### 3.5. Statistical Significance

The exact equality in mean fitness and standard deviation values for Hill Climbing, Simulated Annealing, and GA_Config_4 is a notable result. This suggests that, for this specific problem, these three algorithms have statistically equivalent performances in terms of solution quality, despite their different algorithmic approaches.

To confirm this observation, formal statistical tests, such as Student's t-test to compare means or the F-test to compare variances, could be performed. However, the exact equality of the values is already a strong indicator that there is no statistically significant difference between these three algorithms in terms of solution quality.

## 4. Conclusions

### 4.1. Most Efficient Algorithm

Based on the results of the 30 runs, **Hill Climbing** emerges as the most efficient algorithm for this specific problem. It achieves the same solution quality as Simulated Annealing and GA_Config_4 but in a fraction of the time. Its simplicity and efficiency make it the ideal choice when computation time is a critical factor.

### 4.2. Most Consistent Algorithm

In terms of consistency, **Hill Climbing**, **Simulated Annealing**, and **GA_Config_4** are equally consistent, with the same standard deviation of fitness. However, considering the balance between consistency and execution time, **Hill Climbing** again stands out as the most efficient choice.

### 4.3. Best Balance between Quality and Efficiency

Considering the balance between solution quality, consistency, and computational efficiency, **Hill Climbing** is clearly the most suitable algorithm for this specific problem. It achieves the same solution quality as more complex algorithms but with a significantly lower computational cost.

## 5. Recommendations for Future Work

### 5.1. Parameter Exploration

An interesting direction for future work would be to explore different values for algorithm parameters, such as mutation and crossover rates for Genetic Algorithms, or different cooling schedules for Simulated Annealing. This could potentially further improve the performance of these algorithms.

### 5.2. Algorithm Hybridization

Another promising approach would be algorithm hybridization, combining the strengths of different methods. For example, Hill Climbing could be used to refine solutions generated by a Genetic Algorithm, potentially achieving better results in less time.

### 5.3. Scalability

It would also be interesting to investigate the scalability of these algorithms for larger problems, with more teams or more players. This could reveal performance differences that are not apparent in the current problem.

## 6. Final Summary

This detailed statistical analysis of 30 runs of the four most promising algorithms revealed that Hill Climbing, Simulated Annealing, and GA_Config_4 have statistically equivalent performances in terms of solution quality, but with significant differences in execution time. Hill Climbing emerges as the most efficient algorithm, achieving the same solution quality as more complex algorithms but in a fraction of the time.

These results provide valuable insights for selecting algorithms for the problem of allocating players to sports teams and can inform decisions in similar constrained combinatorial optimization problems.

