import matplotlib.pyplot as plt
import seaborn as sns

def draw_2D_fitness_landscape(ordered_solution_space):
    fitness_values = [solution.fitness() for solution in ordered_solution_space]

    x_labels = [repr(solution) for solution in ordered_solution_space]

    sns.lineplot(x=x_labels, y=fitness_values)

    plt.ylabel('Fitness')
    plt.xlabel('Solutions')
    plt.show()