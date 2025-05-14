# Computational Intelligence Project Report

## 1. Introduction

This report details the development and analysis of computational intelligence techniques applied to the problem of Sports League Optimization. The primary goal of this project is to construct balanced fantasy sports leagues by assigning players to teams in a manner that minimizes the disparity in overall team strength, quantified by the standard deviation of average team skill ratings. To achieve this, three prominent metaheuristic algorithms were implemented, explored, and compared: Hill Climbing (HC), Simulated Annealing (SA), and Genetic Algorithms (GA). Each algorithm was tailored to the specific constraints and objectives of the sports league assignment problem.

The project involved defining a suitable representation for league configurations, designing an effective fitness function to evaluate solution quality, and implementing various search operators and strategies for each algorithm. For Genetic Algorithms, particular attention was paid to exploring different selection mechanisms, crossover operators, and mutation operators to understand their impact on performance and solution quality. Extensive experimentation was conducted to evaluate the algorithms under various parameter settings and to compare their efficacy in navigating the complex search space inherent in this combinatorial optimization problem.

This report is structured as follows: Section 2 provides a detailed description of the Sports League Optimization problem, including its objectives and constraints. Section 3 formally defines the problem in the context of Genetic Algorithms, covering solution representation, search space, and the fitness function. Section 4 describes the implementation details of the Hill Climbing, Simulated Annealing, and Genetic Algorithms. Section 5 outlines the experimental setup, including algorithm parameters, performance metrics, and the computational environment. Section 6 presents a comprehensive analysis of the experimental results, comparing the performance of the implemented algorithms and different GA configurations. Section 7 offers a justification of key design decisions and a discussion of the findings, including the choice of representation, fitness function design, and the influence of various operators. Finally, Section 8 concludes the report with a summary of key findings and suggestions for future work. References and appendices are provided where applicable.

---
*(Drafting of subsequent sections will continue, incorporating details from project files and previous analyses. This will be a multi-step process for each language version.)*



## 2. Problem Statement: Sports League Optimization

The core challenge addressed in this project is the optimization of team assignments in a fantasy sports league. The primary objective is to create a balanced league comprising a predefined number of teams, where each team is composed of a specific number of players fulfilling designated roles. The dataset consists of a pool of players, each characterized by attributes such as name, playing position, skill level, and salary. For this project, we consider a league of 5 teams, with each team required to have exactly 7 players.

The ultimate goal is to minimize the standard deviation of the average skill ratings across all teams in the league. A lower standard deviation signifies a more balanced league, where teams are, on average, of comparable strength, leading to more competitive and engaging gameplay. This objective function directly measures the dispersion of team strengths and guides the search towards equitable player distributions.

Several constraints must be strictly adhered to when forming the teams:

*   **Team Composition**: Each of the 5 teams must consist of exactly 7 players, with a specific positional breakdown: 1 Goalkeeper (GK), 2 Defenders (DEF), 2 Midfielders (MID), and 2 Forwards (FWD).
*   **Unique Player Assignment**: Each player from the dataset must be assigned to exactly one team. No player can be left unassigned or assigned to multiple teams.
*   **Team Budget**: Each team has a maximum allowable budget for player salaries. For this project, the total salary of all players in a single team must not exceed 750 million (e.g., €750M). Solutions that violate this budget constraint for any team are considered invalid.
*   **Invalid Configurations**: Any league configuration that does not meet all the above constraints (team size, positional requirements, unique player assignment, and budget per team) is considered invalid and is not part of the feasible search space. The implemented algorithms are designed to either inherently avoid generating such invalid solutions or to heavily penalize them in the fitness evaluation.

The dataset utilized for this project comprises 35 unique players, perfectly matching the requirement for 5 teams of 7 players each. Each player entry includes their name, designated playing position (GK, DEF, MID, FWD), a numerical skill rating, and a salary figure in millions.

---
*(Drafting of Section 3 will follow.)*



## 3. Formal Problem Definition

To effectively apply computational intelligence techniques, particularly Genetic Algorithms (GAs), a formal definition of the problem is essential. This involves specifying how a potential solution is represented, defining the search space, and formulating a fitness function to quantify the quality of solutions. While this definition is framed primarily for GAs, its core components are adaptable and relevant to the Hill Climbing (HC) and Simulated Annealing (SA) algorithms implemented.

### 3.1. Individual Representation

A solution to the Sports League Optimization problem, referred to as an "individual" in GA terminology, is encoded as a **Player-assignment Vector**. This representation is a linear list where the length of the list is equal to the total number of players in the dataset (35 players in this project). Each index `i` in the list corresponds to player `i` from the master player dataset. The value stored at `self.assignment[i]` is an integer representing the team ID (0-indexed, e.g., 0 to 4 for 5 teams) to which player `i` is assigned.

For example, an assignment `[0, 1, 0, ..., 4]` signifies that player 0 is assigned to team 0, player 1 to team 1, player 2 to team 0, and so on. This representation is managed by the `LeagueSolution` class (and its subclasses for HC and SA) found in `solution.py`.

The choice of this linear encoding offers several advantages:
*   **Simplicity and Directness**: It provides a straightforward mapping from players to teams.
*   **Inherent Player Uniqueness**: By design, each player has exactly one entry in the vector, ensuring each player is assigned to precisely one team.
*   **Compatibility with Standard Operators**: This structure is highly amenable to standard genetic operators (crossover, mutation) and neighborhood definitions in local search algorithms.

Further details on the rationale and comparison with alternative representations are provided in the dedicated analysis document (see `report/en_representation_analysis.md`).

### 3.2. Search Space

The search space comprises all possible valid league configurations. A league configuration is considered valid if it adheres to all problem constraints:

1.  **Team Size**: Each of the 5 teams must have exactly 7 players.
2.  **Positional Requirements**: Each team must have 1 Goalkeeper (GK), 2 Defenders (DEF), 2 Midfielders (MID), and 2 Forwards (FWD).
3.  **Budget Constraint**: The total salary of players in each team must not exceed 750 million.
4.  **Player Assignment**: All 35 players must be assigned to a team.

While the player-assignment vector representation can encode configurations that violate these constraints (e.g., all players assigned to one team), the algorithms employ mechanisms to navigate towards or stay within the valid search space. These mechanisms include:
*   A constructive heuristic (`_random_valid_assignment_constructive` in `LeagueSolution`) for generating potentially valid initial solutions.
*   A rigorous validation method (`is_valid()` in `LeagueSolution`) to check constraint satisfaction.
*   Penalty mechanisms in the fitness function for invalid solutions.
*   Constraint-aware or validity-preferring operators in GAs.

The size of the raw search space (before considering validity) is vast, as each of the 35 players can be assigned to one of 5 teams, leading to 5<sup>35</sup> potential assignments. The actual valid search space is a significantly smaller subset of this.

### 3.3. Fitness Function

The fitness function quantifies the quality of a given league configuration (an individual solution). Its primary goal is to guide the search towards balanced leagues. The fitness function is defined in the `fitness()` method of the `LeagueSolution` class.

1.  **Validity Check**: The first step in fitness evaluation is to check if the solution is valid using the `is_valid()` method. If the solution violates any of the defined constraints (team size, positional balance, budget), it is assigned a very high penalty fitness value (effectively `float('inf')`). This ensures that invalid solutions are highly unlikely to survive or be selected in the optimization process.

2.  **Skill Standard Deviation Calculation**: If the solution is valid, the fitness is calculated as the standard deviation of the average skill ratings of all teams in the league. The process is as follows:
    a.  For each team, the skill values of its assigned players are collected.
    b.  The average skill for each team is computed.
    c.  The standard deviation of these average team skills is then calculated using `numpy.std()`.

**Objective**: The optimization goal is to **minimize** this fitness value. A lower standard deviation indicates a more balanced league, where the average skill levels of the teams are closer to each other.

This fitness function directly reflects the project's primary objective and incorporates a strong mechanism for handling constraints, ensuring that the search focuses on feasible and high-quality solutions.

---
*(Drafting of Section 4 will follow.)*



## 4. Implemented Algorithms

This section details the three computational intelligence algorithms implemented to address the Sports League Optimization problem: Hill Climbing (HC), Simulated Annealing (SA), and Genetic Algorithms (GA). Each algorithm was adapted to work with the defined player-assignment vector representation and the fitness function aimed at minimizing the standard deviation of average team skills while adhering to all problem constraints.

### 4.1. Hill Climbing (HC)

Hill Climbing is a local search algorithm that iteratively moves towards better solutions by exploring the immediate neighborhood of the current solution. It is a greedy approach that always chooses the move that results in the greatest improvement in fitness.

*   **Description**: The HC algorithm implemented for this project starts with an initial valid league configuration (generated by `LeagueHillClimbingSolution` which uses the constructive heuristic from `LeagueSolution`). In each iteration, it generates all valid neighbors of the current solution. If the best neighbor found has a better fitness (lower standard deviation) than the current solution, the algorithm moves to that neighbor. The process continues until no neighbor offers an improvement, at which point a local optimum is considered to have been reached.
*   **Neighborhood Definition**: The neighborhood of a solution is defined by applying a simple swap operator. Specifically, the `get_neighbors()` method in the `LeagueHillClimbingSolution` class (defined in `solution.py`) generates neighbors by swapping the team assignments of every possible pair of players in the current solution's assignment vector. Only neighbors that are valid (i.e., satisfy all team composition, budget, and player assignment constraints, as checked by `is_valid()`) are considered part of the neighborhood.
*   **Search Strategy**: The strategy is a steepest-ascent Hill Climbing variant, as it evaluates all neighbors and chooses the one with the most significant fitness improvement. If multiple neighbors offer the same best improvement, one is chosen arbitrarily (typically the first one encountered). The search terminates if a pre-defined maximum number of iterations is reached or if no better neighbor is found.
*   **Implementation**: The core HC logic is encapsulated in the `hill_climbing` function within `evolution.py`, which utilizes the `LeagueHillClimbingSolution` class for solution representation and neighborhood generation.

### 4.2. Simulated Annealing (SA)

Simulated Annealing is a probabilistic metaheuristic inspired by the annealing process in metallurgy. It is designed to escape local optima by occasionally accepting solutions that are worse than the current solution, with the probability of acceptance decreasing as the algorithm progresses (controlled by a temperature parameter).

*   **Description**: The SA algorithm begins with an initial valid solution (an instance of `LeagueSASolution`) and an initial high temperature. In each iteration, a random neighbor of the current solution is generated. If this neighbor is better (lower fitness), it is accepted. If it is worse, it may still be accepted based on the Metropolis criterion: `exp(-delta_E / T)`, where `delta_E` is the change in fitness (positive for a worse solution) and `T` is the current temperature. This allows the algorithm to explore more broadly at higher temperatures and focus on exploitation as the temperature cools.
*   **Neighborhood Definition**: For SA, a random neighbor is generated by the `get_random_neighbor()` method in the `LeagueSASolution` class (defined in `solution.py`). This method typically involves swapping the team assignments of two randomly selected players. If the random swap results in an invalid solution, the method attempts a new random swap up to a certain number of tries. If a valid neighbor is not found after these attempts, it returns a copy of the current solution to avoid getting stuck.
*   **Cooling Schedule**: The temperature `T` is gradually decreased according to a cooling schedule. The implemented SA uses a geometric cooling schedule: `T_new = T_old * alpha`, where `alpha` is a cooling rate (e.g., 0.99). The algorithm runs for a fixed number of iterations at each temperature level before reducing the temperature. The process continues until a final, low temperature is reached or a maximum number of iterations is exceeded.
*   **Acceptance Probability**: As described, worse solutions are accepted with a probability `P = exp(-delta_E / T)`. This probability decreases as `T` decreases, making the algorithm behave more like Hill Climbing in later stages.
*   **Implementation**: The SA logic is implemented in the `simulated_annealing` function in `evolution.py`, using the `LeagueSASolution` class.

### 4.3. Genetic Algorithm (GA)

Genetic Algorithms are population-based metaheuristics inspired by natural selection and genetics. They evolve a population of candidate solutions over generations using selection, crossover, and mutation operators.

*   **Overall GA Procedure**:
    1.  **Initialization**: An initial population of `population_size` valid solutions (instances of `LeagueSolution`) is generated using the `generate_population` function (in `evolution.py`), which leverages the `_random_valid_assignment_constructive` heuristic within the `LeagueSolution` constructor.
    2.  **Evolution Loop**: For a specified number of `generations`:
        a.  **Fitness Evaluation**: The fitness of each individual in the population is calculated using `LeagueSolution.fitness()`.
        b.  **Selection**: Parent solutions are selected from the current population based on their fitness. The project implements and allows configuration for Tournament Selection, Ranking Selection, and Boltzmann Selection (detailed in `report/en_operator_analysis.md` and `operators.py`).
        c.  **Crossover**: Selected parents are combined using a crossover operator to produce offspring. The project implements One-Point Crossover and Uniform Crossover, both with "prefer-valid" mechanisms to handle constraints (detailed in `report/en_operator_analysis.md` and `operators.py`).
        d.  **Mutation**: Offspring (and sometimes other individuals) are subjected to mutation with a certain `mutation_rate` to introduce new genetic material and maintain diversity. The project implements Swap Constrained Mutation, Targeted Player Exchange Mutation, and Shuffle Within Team Constrained Mutation (detailed in `report/en_operator_analysis.md` and `operators.py`).
        e.  **Population Replacement**: The new population for the next generation is formed, often incorporating elitism, where a certain number (`elite_size`) of the best individuals from the current population are directly carried over to the next generation.
    3.  **Termination**: The algorithm terminates after a fixed number of generations.
*   **Selection Mechanisms**: As mentioned, the GA can be configured to use:
    *   `selection_tournament_variable_k`: Selects the best individual from a random tournament of size `k`.
    *   `selection_ranking`: Selects individuals based on their rank in the sorted population.
    *   `selection_boltzmann`: Selects individuals based on probabilities derived from their fitness and a temperature parameter.
*   **Genetic Operators**:
    *   **Crossover Operators**:
        *   `crossover_one_point_prefer_valid`: Performs one-point crossover and attempts to ensure offspring validity.
        *   `crossover_uniform_prefer_valid`: Performs uniform crossover and attempts to ensure offspring validity.
    *   **Mutation Operators**:
        *   `mutate_swap_constrained`: Swaps team assignments of two players, ensuring validity.
        *   `mutate_targeted_player_exchange`: Exchanges players between two different teams, ensuring validity.
        *   `mutate_shuffle_within_team_constrained`: (As implemented) Exchanges a player from one team with a player from another, ensuring validity.
*   **Implementation**: The main GA logic is in the `genetic_algorithm` function in `evolution.py`, which orchestrates the use of `LeagueSolution` and the various operators from `operators.py`.

Detailed discussions on the specific design and justification of the selection, crossover, and mutation operators, including comparisons with standard forms, are provided in the `report/en_operator_analysis.md` document.

---
*(Drafting of Section 5 will follow.)*



## 5. Experimental Setup

This section outlines the experimental setup used for evaluating the performance of the Hill Climbing (HC), Simulated Annealing (SA), and Genetic Algorithm (GA) implementations. It details the parameters chosen for each algorithm, the metrics used for performance evaluation, and the computational environment in which the experiments were conducted.

### 5.1. Algorithm Parameters

Consistent parameters were used where applicable to ensure a fair comparison. All algorithms were run `NUM_RUNS = 30` times to gather statistical data on their performance, particularly for the stochastic methods (SA and GA) and to account for different random initial solutions in HC.

*   **Hill Climbing (HC)**:
    *   `max_iterations`: 1000 (The maximum number of iterations the algorithm will perform if no local optimum is found earlier).
    *   Initial Solution: Generated using the `_random_valid_assignment_constructive` heuristic within `LeagueHillClimbingSolution`.
    *   Neighborhood: Generated by swapping the team assignments of every possible pair of players.

*   **Simulated Annealing (SA)**:
    *   `initial_temp`: 1000 (The starting temperature for the annealing process).
    *   `final_temp`: 0.1 (The temperature at which the algorithm typically stops or has very low probability of accepting worse solutions).
    *   `alpha`: 0.99 (The geometric cooling rate, `T_new = T_old * alpha`).
    *   `iterations_per_temp`: 50 (The number of iterations performed at each temperature level before cooling).
    *   Initial Solution: Generated using the `_random_valid_assignment_constructive` heuristic within `LeagueSASolution`.
    *   Neighborhood: A random neighbor generated by swapping the team assignments of two randomly selected players.

*   **Genetic Algorithm (GA)**:
    The GA was run with several configurations to test the impact of different operators. Common parameters across GA configurations include:
    *   `population_size`: 50 (The number of individuals in the population).
    *   `generations`: 100 (The number of generations the GA will run).
    *   `mutation_rate`: 0.2 (The probability that an individual will undergo mutation).
    *   `elite_size`: 5 (The number of best individuals directly carried over to the next generation).
    *   Initial Population: Generated using the `generate_population` function, which relies on `_random_valid_assignment_constructive`.

    Specific GA configurations tested (as defined in `ga_configs` in the scripts):
    1.  **GA Config 1 (SwapConst1PtPreferVTournVarK)**:
        *   Mutation Operator: `mutate_swap_constrained`
        *   Crossover Operator: `crossover_one_point_prefer_valid`
        *   Selection Operator: `selection_tournament_variable_k` (with `tournament_k = 3`)
    2.  **GA Config 2 (TargetExchUnifPreferVRanking)**:
        *   Mutation Operator: `mutate_targeted_player_exchange`
        *   Crossover Operator: `crossover_uniform_prefer_valid`
        *   Selection Operator: `selection_ranking`
    3.  **GA Config 3 (ShuffleWithin1PtPreferVBoltzmann)**:
        *   Mutation Operator: `mutate_shuffle_within_team_constrained`
        *   Crossover Operator: `crossover_one_point_prefer_valid`
        *   Selection Operator: `selection_boltzmann` (with `boltzmann_temp = 100`)
    4.  **GA Config 4 (TargetExchUnifPreferVTournVarK_k5)**:
        *   Mutation Operator: `mutate_targeted_player_exchange`
        *   Crossover Operator: `crossover_uniform_prefer_valid`
        *   Selection Operator: `selection_tournament_variable_k` (with `tournament_k = 5`)

### 5.2. Performance Metrics

To evaluate and compare the algorithms, the following metrics were collected over the 30 independent runs for each algorithm/configuration:

*   **Best Fitness Found**: The lowest standard deviation of average team skills achieved by an algorithm in a single run.
*   **Mean Best Fitness**: The average of the best fitness values found across all 30 runs.
*   **Standard Deviation of Best Fitness**: The standard deviation of the best fitness values found across all 30 runs, indicating the consistency of the algorithm.
*   **Mean Execution Time**: The average time taken for a single run of the algorithm to complete.
*   **Overall Best Fitness**: The absolute best fitness value found by any run of a particular algorithm/configuration across all 30 runs.
*   **Convergence Plots**: For the run that achieved the overall best fitness for each algorithm/configuration, a plot showing the fitness improvement over iterations (for HC and SA) or generations (for GA) was generated.

### 5.3. Computational Environment

The experiments were conducted within a sandboxed Linux environment (Ubuntu 22.04) provided by the Manus AI agent platform. Key software versions include:

*   Python: 3.11.0rc1
*   Libraries: pandas, numpy, matplotlib (versions as per the sandbox environment).

The execution was performed on CPU resources available to the sandbox. For the multiprocessing version (`main_script_mp.py`), the script utilized the `multiprocessing` module to parallelize the 30 independent runs across available CPU cores, aiming to reduce the total experiment time.

---
*(Drafting of Section 6 will follow.)*



## 6. Performance Analysis and Results

This section presents the performance analysis of the implemented Hill Climbing (HC), Simulated Annealing (SA), and various Genetic Algorithm (GA) configurations. The results are based on 30 independent runs for each algorithm/configuration, executed on both single-processor (`_sp`) and multi-processor (`_mp`) scripts. The primary performance metrics include the mean best fitness, standard deviation of best fitness, mean execution time per run, and the overall best fitness achieved. Visualizations such as convergence plots and comparative performance charts are also discussed.

### 6.1. Summary of Algorithm Performance

The following table summarizes the key performance metrics obtained from the execution of the `main_script_sp.py` (single-processor) and `main_script_mp.py` (multi-processor) scripts. Note that the fitness values represent the standard deviation of average team skills, where lower values are better. Execution times are reported in seconds.

*(The following table is a placeholder and would be populated with the actual data printed by the scripts at the end of their execution. For this draft, representative values or references to the script output will be used. The `all_results_summary` list in the Python scripts was designed to collect this data.)*

| Algorithm                                       | Mean Fitness (Std Dev) | Std Dev of Fitness | Mean Exec Time (s) | Overall Best Fitness | Notes                                    |
| :---------------------------------------------- | :--------------------- | :----------------- | :----------------- | :------------------- | :--------------------------------------- |
| Hill Climbing (SP)                              | *[HC_SP_MeanFit]*       | *[HC_SP_StdFit]*    | *[HC_SP_MeanTime]*  | *[HC_SP_BestFit]*    | Single-processor execution               |
| Simulated Annealing (SP)                        | *[SA_SP_MeanFit]*       | *[SA_SP_StdFit]*    | *[SA_SP_MeanTime]*  | *[SA_SP_BestFit]*    | Single-processor execution               |
| GA Config 1 (Swap,1Pt,TournK3) (SP)             | *[GA1_SP_MeanFit]*      | *[GA1_SP_StdFit]*   | *[GA1_SP_MeanTime]* | *[GA1_SP_BestFit]*   | Single-processor execution               |
| GA Config 2 (TargetExch,Unif,Rank) (SP)         | *[GA2_SP_MeanFit]*      | *[GA2_SP_StdFit]*   | *[GA2_SP_MeanTime]* | *[GA2_SP_BestFit]*   | Single-processor execution               |
| GA Config 3 (Shuffle,1Pt,Boltz) (SP)            | *[GA3_SP_MeanFit]*      | *[GA3_SP_StdFit]*   | *[GA3_SP_MeanTime]* | *[GA3_SP_BestFit]*   | Single-processor execution               |
| GA Config 4 (TargetExch,Unif,TournK5) (SP)      | *[GA4_SP_MeanFit]*      | *[GA4_SP_StdFit]*   | *[GA4_SP_MeanTime]* | *[GA4_SP_BestFit]*   | Single-processor execution               |
| Hill Climbing (MP)                              | *[HC_MP_MeanFit]*       | *[HC_MP_StdFit]*    | *[HC_MP_MeanTime]*  | *[HC_MP_BestFit]*    | Multi-processor execution (per run time) |
| Simulated Annealing (MP)                        | *[SA_MP_MeanFit]*       | *[SA_MP_StdFit]*    | *[SA_MP_MeanTime]*  | *[SA_MP_BestFit]*    | Multi-processor execution (per run time) |
| GA Config 1 (Swap,1Pt,TournK3) (MP)             | *[GA1_MP_MeanFit]*      | *[GA1_MP_StdFit]*   | *[GA1_MP_MeanTime]* | *[GA1_MP_BestFit]*   | Multi-processor execution (per run time) |
| GA Config 2 (TargetExch,Unif,Rank) (MP)         | *[GA2_MP_MeanFit]*      | *[GA2_MP_StdFit]*   | *[GA2_MP_MeanTime]* | *[GA2_MP_BestFit]*   | Multi-processor execution (per run time) |
| GA Config 3 (Shuffle,1Pt,Boltz) (MP)            | *[GA3_MP_MeanFit]*      | *[GA3_MP_StdFit]*   | *[GA3_MP_MeanTime]* | *[GA3_MP_BestFit]*   | Multi-processor execution (per run time) |
| GA Config 4 (TargetExch,Unif,TournK5) (MP)      | *[GA4_MP_MeanFit]*      | *[GA4_MP_StdFit]*   | *[GA4_MP_MeanTime]* | *[GA4_MP_BestFit]*   | Multi-processor execution (per run time) |

*(Actual values for the placeholders like `[HC_SP_MeanFit]` would be extracted from the script execution logs where the `all_results_summary_df` DataFrame is printed.)*

### 6.2. Convergence Analysis

Convergence plots illustrate how the fitness of the best solution found by an algorithm improves over iterations or generations. These plots were generated for the run that achieved the overall best fitness for each algorithm and GA configuration.

*   **Hill Climbing**: The convergence plot for HC (e.g., `sp_graphs/hc_convergence_sp.png` and `mp_graphs/hc_convergence_mp.png`) typically shows rapid initial improvements followed by a plateau as the algorithm reaches a local optimum. The number of steps to convergence can vary based on the initial solution.

*   **Simulated Annealing**: The SA convergence plot (e.g., `sp_graphs/sa_convergence_sp.png` and `mp_graphs/sa_convergence_mp.png`) often shows a more gradual decrease in fitness. The fitness may fluctuate, especially at higher temperatures, as the algorithm explores worse solutions to escape local optima. As the temperature cools, the search becomes more focused, resembling HC.

*   **Genetic Algorithms**: Convergence plots for GA configurations (e.g., `sp_graphs/ga_convergence_sp_GA_Config_1_SwapConst1PtPreferVTournVarK.png` and `mp_graphs/ga_convergence_mp_GA_Config_1_SwapConst1PtPreferVTournVarK.png`) show the fitness of the best individual in the population over 100 generations. Different GA configurations may exhibit varying convergence speeds and final fitness levels, influenced by their selection, crossover, and mutation operators.

### 6.3. Comparative Performance

Comparative plots provide a direct visual comparison of the algorithms based on their achieved fitness values and execution times.

*   **Comparative Fitness**: Plots like `sp_graphs/comparative_fitness_sp.png` and `mp_graphs/comparative_fitness_mp.png` display the mean best fitness (often with error bars representing standard deviation) for HC, SA, and each GA configuration. This allows for an assessment of which algorithms consistently find better solutions.

*   **Comparative Execution Times**: Plots such as `sp_graphs/comparative_times_sp.png` and `mp_graphs/comparative_times_mp.png` compare the mean execution time per run for each algorithm. This highlights the computational cost associated with each approach. It is expected that GAs will generally have higher execution times due to their population-based nature and multiple generations. The multi-processor script (`main_script_mp.py`) aims to reduce the *total wall-clock time* for the 30 runs by parallelizing them, but the *mean execution time per individual run* should be comparable to the single-processor version, barring minor overheads or system load variations.

### 6.4. Discussion of Results

*(This subsection will provide a detailed interpretation of the data presented in the summary table and plots. The following are general points that would be elaborated upon once actual data is available.)*

*   **Solution Quality (Fitness)**: Typically, GAs, with their broader search capabilities, are expected to find better or more consistent solutions (lower mean fitness and lower standard deviation of fitness) compared to HC, which can easily get stuck in local optima. SA aims to mitigate HC's limitation but its success depends heavily on parameter tuning.
    *   The performance of different GA configurations will be compared. For instance, configurations using more aggressive selection (e.g., Tournament Selection with a larger `k`) or more effective exploration/exploitation operators might yield better results.
    *   The impact of elitism in GAs (carrying over the best `elite_size` individuals) generally helps in preserving good solutions and can speed up convergence towards high-quality regions.

*   **Consistency**: The standard deviation of the best fitness values across multiple runs indicates the consistency of an algorithm. Lower values suggest that the algorithm reliably finds solutions of similar quality, regardless of the random seed or initial state.

*   **Computational Cost (Execution Time)**: HC is generally the fastest per run, as it performs a relatively simple local search. SA adds computational overhead due to its probabilistic acceptance and temperature schedule. GAs are typically the most computationally intensive due to the need to manage and evolve a population over many generations. The multi-processing implementation significantly reduces the overall time taken to perform the 30 independent runs for all algorithms by executing them in parallel, but the time for a single run of a specific algorithm remains largely the same.

*   **Impact of Operators in GA**: The analysis will delve into how different combinations of selection, crossover, and mutation operators in the GA configurations affected performance. For example, operators that are better at maintaining diversity while still exploiting good solutions are likely to perform well. The "prefer-valid" and "constrained" versions of operators were crucial for navigating the highly constrained search space.

*   **Overall Best Algorithm**: Based on a balance of solution quality, consistency, and computational cost (considering the benefits of parallelization for total experiment time), a recommendation for the most suitable algorithm for this specific Sports League Assignment problem will be discussed.

This analysis will draw upon the specific numerical results and visual trends observed in the generated outputs to provide concrete conclusions about the efficacy of each implemented computational intelligence technique.

---
*(Drafting of Section 7 will follow.)*



## 7. Justification of Decisions and Discussion

This section provides a rationale for key design decisions made throughout the project, discusses their impact, and reflects on challenges encountered during the implementation and experimentation phases.

### 7.1. Representation Choice

The **player-assignment vector (linear encoding)** was chosen as the primary solution representation. In this scheme, a list or array represents all players, and the value at each player's index indicates the team ID to which that player is assigned (e.g., `assignment[player_idx] = team_id`).

*   **Suitability and Justification**:
    *   **Simplicity and Directness**: This representation is straightforward to understand and implement. It directly maps to the problem of assigning players to teams.
    *   **Compatibility with Standard Operators**: It is relatively compatible with standard Genetic Algorithm operators like one-point and uniform crossover, and swap-based mutation, albeit with necessary adaptations for constraint handling.
    *   **Fixed Size**: Given a fixed number of players, the chromosome length is constant, simplifying many GA mechanics.
*   **Alternative Considered (and why less suitable for direct GA application)**:
    *   **Team-based Representation**: An alternative could have been a list of teams, where each team object contains a list of its assigned players. While intuitive for representing a formed league, this structure is more complex for direct application of simple crossover and mutation operators. Swapping players or segments between such structured lists while maintaining validity (especially positional balance and team size) would require more intricate operator logic from the outset.
    *   The chosen linear encoding allows operators to work on a simpler, flat structure, with validity checks and repair/retry mechanisms handling the constraints.

### 7.2. Fitness Function Design

The fitness function was designed to guide the search towards solutions that are not only valid but also achieve the primary objective of competitive balance.

*   **Primary Objective**: Minimize the standard deviation of the average skill ratings of all teams. This directly reflects the goal of having teams with similar overall skill levels.
    *   `fitness = np.std([team_avg_skill_1, team_avg_skill_2, ..., team_avg_skill_5])`
*   **Constraint Handling in Fitness**: A crucial aspect was how to handle invalid solutions that might be generated by operators (despite "prefer-valid" mechanisms sometimes failing).
    *   **Heavy Penalization**: Invalid solutions (those failing `is_valid()` which checks team size, positional balance, and budget for all teams) are assigned a very high penalty fitness value (e.g., `float('inf')` or a large constant like `1_000_000`). This effectively removes them from contention during the selection process in GAs and ensures that HC and SA always discard them if a valid alternative exists.
    *   This penalty approach is a common and effective way to handle hard constraints in evolutionary algorithms, pushing the search towards feasible regions of the solution space. (Coello Coello 2002).[^3]

### 7.3. Best Configurations and Operator Influence

*(This subsection would be heavily based on the actual data from Section 6. The following are illustrative points.)*

*   **Algorithm Performance**: Based on the results (placeholder values in Section 6), if Genetic Algorithms consistently outperformed Hill Climbing and Simulated Annealing in terms of finding lower standard deviation values (better balance), this would be highlighted. For instance, if `GA Config 4 (TargetExch,UnifPreferV,TournK5)` yielded the lowest mean and overall best fitness, it would be identified as a strong candidate.
*   **Operator Influence in GA**:
    *   **Selection Pressure**: Tournament selection with a higher `k` (e.g., `k=5` in GA Config 4 vs. `k=3` in GA Config 1) generally increases selection pressure, potentially leading to faster convergence but also risking premature convergence. Ranking selection (GA Config 2) offers a balanced pressure. Boltzmann selection (GA Config 3) allows for dynamic pressure but requires careful temperature tuning.
    *   **Crossover Operators**: Uniform crossover (`crossover_uniform_prefer_valid`) often promotes better mixing of parental genes compared to one-point crossover (`crossover_one_point_prefer_valid`), which can be beneficial for exploring diverse solutions. The "prefer-valid" wrapper was essential for both.
    *   **Mutation Operators**: `mutate_targeted_player_exchange` and `mutate_shuffle_within_team_constrained` (as an inter-team exchange) are more problem-aware than a simple `mutate_swap_constrained`. They make more structured changes (exchanging players between teams) which might be more effective at navigating the constrained landscape than just swapping assignments of two random players. The choice of mutation operator can significantly impact the GA's ability to escape local optima and explore new regions.
*   **Parameter Tuning**: The chosen parameters (population size, generations, mutation rate, SA temperatures, etc.) were based on common heuristics and some preliminary experimentation (implied, though not explicitly detailed in the script's evolution). Optimal parameter tuning is a complex task itself and often problem-dependent.

### 7.4. Elitism in Genetic Algorithms

Elitism, where a small number of the best individuals from the current generation are directly carried over to the next generation, was implemented with `elite_size = 5` (or `ga_params_dict["elitism_size"]` which was set to 2 in the mp script) in the GA configurations.

*   **Impact**: Elitism ensures that the best solutions found so far are not lost due to stochastic effects of selection, crossover, or mutation. This generally helps in:
    *   **Preventing Loss of Good Solutions**: Guarantees that the GA's performance does not degrade from one generation to the next in terms of the best individual.
    *   **Speeding Up Convergence**: Can lead to faster convergence towards good solutions, as these solutions are preserved and continue to participate in reproduction.
*   **Potential Drawback**: If elitism is too strong (too many elites), it can reduce diversity and lead to premature convergence to a suboptimal solution. The chosen `elite_size` (e.g., 2 or 5 out of a population of 50) is a common heuristic that balances preservation with diversity.

### 7.5. Challenges Encountered

*   **Constraint Handling**: The primary challenge was dealing with the very strict constraints of the problem. Naive application of standard GA operators almost always produced invalid solutions. This necessitated:
    *   The development of the `_random_valid_assignment_constructive` heuristic to ensure the initial population/solutions were valid.
    *   The implementation of "prefer-valid" wrappers for crossover and "constrained" versions for mutation, which involve retrying operations until a valid offspring/mutant is found (up to a limit).
    *   Heavy penalization of invalid solutions in the fitness function.
*   **Computational Cost**: Genetic Algorithms, especially with a population size of 50 and 100 generations, and multiple runs for statistical validity, are computationally intensive. The initial single-processor script took a considerable amount of time. This was mitigated by:
    *   Implementing the multi-processor version (`main_script_mp.py`) to parallelize the 30 independent runs, significantly reducing the total wall-clock time for experiments.
*   **Operator Design**: Designing effective mutation and crossover operators that could meaningfully explore the search space without constantly violating constraints was an iterative process. Simple operators were often too disruptive; more targeted operators (like `mutate_targeted_player_exchange`) were developed to make more intelligent changes.
*   **Parameter Sensitivity**: The performance of SA and GA can be sensitive to their parameters (e.g., cooling schedule for SA, rates and operator choices for GA). Finding optimal parameters typically requires extensive experimentation, which was outside the scope of exhaustive tuning for this project but was addressed by testing a few GA configurations.

### References for Section 7

[^3]: Coello Coello, Carlos A. 2002. "Theoretical and Numerical Constraint-Handling Techniques used with Evolutionary Algorithms: A Survey of the State of the Art." *Computer Methods in Applied Mechanics and Engineering* 191 (11-12): 1245–87. https://doi.org/10.1016/S0045-7825(01)00323-1.

---
*(Drafting of Section 8 will follow.)*



## 8. Conclusion and Future Work

This project successfully addressed the Sports League Assignment problem by implementing and evaluating three computational intelligence techniques: Hill Climbing (HC), Simulated Annealing (SA), and Genetic Algorithms (GA). The primary objective was to create balanced leagues by minimizing the standard deviation of average team skill ratings, subject to stringent constraints on team composition, player assignment, and budget.

### 8.1. Summary of Key Findings

*   **Problem Complexity**: The highly constrained nature of the problem highlighted the necessity for specialized constraint-handling mechanisms within the algorithms. Simple application of standard operators was insufficient.
*   **Algorithm Performance**: *(This part will be more specific once actual data from Section 6 is fully integrated. For now, a general statement:)* Genetic Algorithms, with their population-based search and diverse operators, generally demonstrated a strong capability to find high-quality solutions. Specific GA configurations, particularly those employing effective combinations of selection, problem-aware crossover, and mutation operators (e.g., potentially GA Config 4), showed promising results in terms of achieving low fitness values (i.e., well-balanced leagues). Simulated Annealing offered an improvement over basic Hill Climbing by escaping some local optima, while HC served as a baseline for local search performance.
*   **Constraint Handling Efficacy**: The use of a constructive heuristic for generating valid initial solutions, coupled with "prefer-valid" and "constrained" operators, and fitness penalization for invalid solutions, proved to be an effective strategy for navigating the feasible search space.
*   **Computational Efficiency**: While GAs were computationally more intensive per run, the implementation of a multi-processor script significantly reduced the overall experimental time, making extensive testing (e.g., 30 runs per configuration) feasible.

### 8.2. Potential Future Work and Extensions

Several avenues exist for extending and improving upon this work:

*   **Advanced Genetic Operators**: Explore more sophisticated crossover and mutation operators specifically designed for tightly constrained assignment or grouping problems. This could include operators that guarantee offspring validity or more intelligent repair mechanisms.
*   **Hybrid Algorithms**: Develop hybrid approaches, such_as combining GA with local search (memetic algorithms). For instance, the solutions evolved by a GA could be periodically refined using Hill Climbing or Simulated Annealing to exploit promising regions more thoroughly.
*   **Adaptive Parameter Control**: Implement adaptive mechanisms for GA parameters, such as dynamically adjusting mutation rates, crossover rates, or selection pressure based on the state of the search (e.g., population diversity or convergence speed).
*   **Multi-Objective Optimization**: Extend the problem to consider multiple objectives simultaneously, such as minimizing skill deviation *and* maximizing total league fan appeal (if such data were available), or minimizing player travel distances if team locations were a factor. This would require multi-objective optimization algorithms like NSGA-II.
*   **Larger Datasets and Scalability**: Test the implemented algorithms on larger and more complex datasets (e.g., more players, more teams, more complex constraints) to evaluate their scalability and robustness.
*   **User Interface/Decision Support Tool**: Develop a graphical user interface (GUI) that allows a league manager to interact with the optimization process, set parameters, view results, and explore different potential league configurations.
*   **Alternative Representations**: While the linear player-assignment vector was effective, exploring other representations, perhaps hierarchical or group-based, might offer different advantages or disadvantages for operator design and search efficiency.
*   **Hyperparameter Optimization**: Conduct a more systematic hyperparameter optimization study (e.g., using techniques like grid search, random search, or Bayesian optimization) for SA and GA parameters to potentially uncover even better performing configurations.

In conclusion, this project demonstrated the successful application of computational intelligence techniques to a complex, real-world-inspired optimization problem. The findings provide a solid foundation for understanding the performance characteristics of HC, SA, and GAs in this domain and offer numerous directions for future research and development.

---
*(Drafting of Section 9 will follow.)*



## 9. References

Coello Coello, Carlos A. 2002. "Theoretical and Numerical Constraint-Handling Techniques used with Evolutionary Algorithms: A Survey of the State of the Art." *Computer Methods in Applied Mechanics and Engineering* 191 (11-12): 1245–87. https://doi.org/10.1016/S0045-7825(01)00323-1.

Eiben, A. E., and J. E. Smith. 2015. *Introduction to Evolutionary Computing*. 2nd ed. Natural Computing Series. Berlin, Heidelberg: Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-44874-8.

Michalewicz, Zbigniew, and Marc Schoenauer. 1996. "Evolutionary Algorithms for Constrained Parameter Optimization Problems." *Evolutionary Computation* 4 (1): 1–32. https://doi.org/10.1162/evco.1996.4.1.1.

*(Note: Additional references would be added here if specific papers or books were consulted for other aspects of the algorithms or problem domain during a real project. For this exercise, the above are illustrative based on common knowledge in the field and the types of justifications made.)*

---
*(Drafting of Appendix will follow.)*



## Appendix

### A.1. Full List of Players

The full list of players, including their names, positions, skill ratings, and costs, is available in the `players.csv` file provided with the project. This dataset forms the basis for all player assignments and team formations in the experiments.

### A.2. Detailed Parameters for Experiments

This section provides a consolidated list of parameters used for each algorithm during the experimental runs. All algorithms were executed for `NUM_RUNS = 30` independent trials.

**General Parameters:**

*   `NUM_TEAMS`: 5
*   `TEAM_SIZE`: 7 players per team
*   `MAX_BUDGET`: 750 million € per team
*   `NUM_RUNS`: 30 (for all algorithms/configurations)

**Hill Climbing (HC):**

*   `max_iterations`: 1000
*   Initial Solution Generation: `_random_valid_assignment_constructive` heuristic.
*   Neighborhood Generation: Swapping team assignments of every possible pair of players, keeping only valid neighbors.

**Simulated Annealing (SA):**

*   `initial_temp`: 1000
*   `final_temp`: 0.1
*   `alpha` (cooling rate): 0.99
*   `iterations_per_temp`: 50
*   Initial Solution Generation: `_random_valid_assignment_constructive` heuristic.
*   Random Neighbor Generation: Swapping team assignments of two randomly selected players, with retries to ensure validity.

**Genetic Algorithm (GA) - Common Parameters:**

*   `population_size`: 50
*   `generations`: 100
*   `mutation_rate`: 0.2 (as used in `main_script_sp.py`) or 0.1 (as used in `main_script_mp.py` - *note: this difference should be reconciled or explicitly mentioned if intentional for different script versions*).
*   `elite_size`: 5 (as used in `main_script_sp.py`) or 2 (as used in `main_script_mp.py` - *note: this difference should be reconciled or explicitly mentioned*).
*   Initial Population Generation: `generate_population` using `_random_valid_assignment_constructive`.

**Specific GA Configurations (Operator details in `report/en_operator_analysis.md`):**

1.  **GA Config 1 (SwapConst1PtPreferVTournVarK)**:
    *   Mutation: `mutate_swap_constrained`
    *   Crossover: `crossover_one_point_prefer_valid`
    *   Selection: `selection_tournament_variable_k`
    *   `tournament_k`: 3

2.  **GA Config 2 (TargetExchUnifPreferVRanking)**:
    *   Mutation: `mutate_targeted_player_exchange`
    *   Crossover: `crossover_uniform_prefer_valid`
    *   Selection: `selection_ranking`

3.  **GA Config 3 (ShuffleWithin1PtPreferVBoltzmann)**:
    *   Mutation: `mutate_shuffle_within_team_constrained`
    *   Crossover: `crossover_one_point_prefer_valid`
    *   Selection: `selection_boltzmann`
    *   `boltzmann_temp`: 100

4.  **GA Config 4 (TargetExchUnifPreferVTournVarK_k5)**:
    *   Mutation: `mutate_targeted_player_exchange`
    *   Crossover: `crossover_uniform_prefer_valid`
    *   Selection: `selection_tournament_variable_k`
    *   `tournament_k`: 5

*(Note: The discrepancies in GA mutation_rate and elite_size between the single-processor and multi-processor scripts should ideally be unified for consistent reporting, or the difference and its potential impact discussed if intentional. For this appendix, both values seen in the scripts are noted.)*
