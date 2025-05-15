# Multiprocessing Phase Report (5 Runs)

## 1. Introduction

This section of the project focuses on evaluating the performance of optimization algorithms (Hill Climbing, Simulated Annealing, and four configurations of Genetic Algorithms) when executed multiple times in parallel. The main objective of this phase was to obtain statistically more robust data on the effectiveness and efficiency of each approach by running each algorithm 5 times. The use of multiprocessing allowed these concurrent executions, optimizing the total experimentation time.

The results of this phase are crucial for identifying the most promising algorithms and configurations, which will later be subjected to a higher number of runs (30 runs) for an even more in-depth performance analysis.

## 2. Experimental Methodology

### 2.1. Evaluated Algorithms

The following algorithms and configurations were evaluated in this phase:

1.  **Hill Climbing (HC)**: An iterative local search algorithm.
2.  **Simulated Annealing (SA)**: A probabilistic algorithm inspired by the annealing process in metallurgy.
3.  **Genetic Algorithms (GA)**: Four distinct configurations were tested, varying mutation, crossover, and selection operators:
    *   **GA_Config_1_SwapConst1PtPreferVTournVarK**: Swap Constrained Mutation, One-Point Prefer Valid Crossover, Tournament Variable K Selection (k=3).
    *   **GA_Config_2_TargetExchUnifPreferVRanking**: Targeted Player Exchange Mutation, Uniform Prefer Valid Crossover, Ranking Selection.
    *   **GA_Config_3_ShuffleWithin1PtPreferVBoltzmann**: Shuffle Within Team Constrained Mutation, One-Point Prefer Valid Crossover, Boltzmann Selection.
    *   **GA_Config_4_TargetExchUnifPreferVTournVarK_k5**: Targeted Player Exchange Mutation, Uniform Prefer Valid Crossover, Tournament Variable K Selection (k=5).

### 2.2. Execution Parameters

*   **Number of Runs per Algorithm**: Each algorithm/configuration was executed 5 times.
*   **Parallelization**: Executions were performed in parallel using Python's `multiprocessing` module, leveraging available CPU cores to speed up the process.
*   **Collected Metrics**: For each run and, subsequently, for the set of 5 runs of each algorithm, the following metrics were recorded:
    *   Overall Best Fitness: The lowest value of the fitness function (standard deviation of average team skills) achieved.
    *   Mean Fitness: The average of the best fitness values obtained in the 5 runs.
    *   Standard Deviation of Fitness: A measure of the algorithm's consistency in achieving good results.
    *   Mean Execution Time: The average time (in seconds) each algorithm took to complete one run.
*   **Environment**: Experiments were conducted in the provided sandbox environment, with player data loaded from the `players.csv` file.

### 2.3. Population Generation and Solution Validity

For Genetic Algorithms, initial population generation and maintenance of solution validity throughout generations followed the same constraints defined in the single-processor phase (team structure, maximum budget, etc.). Particular attention was paid to warnings about difficulty in generating valid individuals, although the script was configured to attempt multiple times to generate valid initial solutions for HC and SA, and GAs have intrinsic mechanisms to handle validity.

## 3. Results and Discussion

After the successful execution of the `main_script_mp.py` script for 5 runs of each algorithm, the results were compiled and analyzed. The following table summarizes the performance of each algorithm/configuration:

```
Algorithm,Mean Fitness,Std Dev Fitness,Mean Exec Time (s),Overall Best Fitness,Mutation Op,Crossover Op,Selection Op
Hill Climbing (MP-5 runs),0.06709518506727775,0.01990465584884448,0.46889443397521974,0.05714285714285552,N/A,N/A,N/A
Simulated Annealing (MP-5 runs),0.05714285714285552,0.0,17.598721027374268,0.05714285714285552,N/A,N/A,N/A
GA: GA_Config_1_SwapConst1PtPreferVTournVarK (MP-5 runs),0.0770475129917,0.024378125167686872,6.0489390850067135,0.05714285714285552,mutate_swap_constrained,crossover_one_point_prefer_valid,selection_tournament_variable_k
GA: GA_Config_2_TargetExchUnifPreferVRanking (MP-5 runs),0.08898633875210829,0.04332839580277237,61.911612796783444,0.05714285714285552,mutate_targeted_player_exchange,crossover_uniform_prefer_valid,selection_ranking
GA: GA_Config_3_ShuffleWithin1PtPreferVBoltzmann (MP-5 runs),0.0936131100520787,0.032132600170602175,61.22301082611084,0.05714285714285552,mutate_shuffle_within_team_constrained,crossover_one_point_prefer_valid,selection_boltzmann
GA: GA_Config_4_TargetExchUnifPreferVTournVarK_k5 (MP-5 runs),0.05714285714285552,0.0,8.188594150543214,0.05714285714285552,mutate_targeted_player_exchange,crossover_uniform_prefer_valid,selection_tournament_variable_k
```

Convergence plots for the best run of each algorithm and comparative plots of execution time and best fitness were also generated and saved in `/home/ubuntu/CIFO_EXTENDED_Project/images_mp/run_5_results/`.

### 3.1. Solution Quality Analysis (Fitness)

A notable result is observed: all algorithms/configurations (Simulated Annealing, GA_Config_1, GA_Config_2, GA_Config_3, and GA_Config_4) managed to achieve the same **Overall Best Fitness** value of approximately **0.057143**. Hill Climbing, although reaching this value in some runs, showed a slightly higher **Mean Fitness** (0.0671), indicating less consistency in reaching the best solution found by the others.

**Simulated Annealing** and the **GA_Config_4_TargetExchUnifPreferVTournVarK_k5** configuration stood out for their consistency, both presenting a **Standard Deviation of Fitness** of **0.0**. This means that in all 5 runs, these two algorithms converged to the same optimal fitness value. The other GA configurations (Config_1, Config_2, Config_3) and HC showed greater variability in fitness results between runs.

The **Mean Fitness** of GA_Config_4 (0.057143) was the best among the GAs, matching SA. GA_Config_1, GA_Config_2, and GA_Config_3 had worse mean fitness values (0.0770, 0.0890, and 0.0936, respectively).

### 3.2. Efficiency Analysis (Execution Time)

Regarding **Mean Execution Time**, the differences are significant:

*   **Hill Climbing** was by far the fastest, with an average time of approximately **0.47 seconds** per run.
*   Among the Genetic Algorithms, **GA_Config_1_SwapConst1PtPreferVTournVarK** was the fastest, at about **6.05 seconds**.
*   **GA_Config_4_TargetExchUnifPreferVTournVarK_k5**, which demonstrated excellent consistency and solution quality, had a mean time of **8.19 seconds**.
*   **Simulated Annealing**, despite its optimal performance in fitness and consistency, was considerably slower, with an average of **17.60 seconds**.
*   The **GA_Config_2** and **GA_Config_3** configurations were the slowest, with mean execution times around **61-62 seconds**, making them less attractive despite having achieved the overall best fitness in some runs.

### 3.3. Additional Observations

*   The ability of all algorithms (except, on average, HC) to find the same best fitness suggests that, for 5 runs and the current parameters, this might be a robust local optimum or even the global optimum for the problem with the given data.
*   Consistency (low standard deviation of fitness) is a desirable characteristic, and SA and GA_Config_4 were exemplary in this aspect.
*   The trade-off between solution quality, consistency, and execution time is a key factor in selecting algorithms for more extensive testing.

## 4. Preliminary Conclusions and Next Steps

Based on the results of 5 multiprocessing runs:

*   **Hill Climbing** is extremely fast but less consistent in finding the best solution compared to others.
*   **Simulated Annealing** consistently finds the best solution but at a moderate execution time cost.
*   The Genetic Algorithm configuration **GA_Config_4_TargetExchUnifPreferVTournVarK_k5** emerges as a very strong candidate, as it matched SA and HC in overall best fitness, demonstrated perfect consistency (std dev = 0), and had a significantly lower execution time than SA and the other two slower GA configurations (Config_2 and Config_3).
*   **GA_Config_1_SwapConst1PtPreferVTournVarK** is also notable for its speed among GAs, albeit with lower consistency in mean fitness.

These results provide a solid basis for the next phase of the project: selecting the most promising algorithms to be run 30 times. A detailed analysis of these results will allow for refining the choice of algorithms and their parameters for future investigations.

The next steps include translating this report into Bengali, followed by the formal identification of the algorithms to be carried forward to the 30-run phase.

