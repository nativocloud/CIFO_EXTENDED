import random

def standard_crossover(parent1_repr, parent2_repr):
    """
    Performs standard one-point crossover on two parent representations.

    This operator selects a random crossover point (not at the edges) and 
    exchanges the tail segments of the two parents to produce two offspring. 
    The crossover point is the same for both parents and ensures at least one 
    gene is inherited from each parent before and after the point.

    Parameters:
        parent1_repr (str or list): The first parent representation.
        parent2_repr (str or list): The second parent representation.
            Both parents must have the same length and type.

    Returns:
        tuple: A pair of offspring representations (offspring1, offspring2), 
        of the same type as the parents.

    Raises:
        ValueError: If parent representations are not the same length.
    """

    if not (isinstance(parent1_repr, list) or isinstance(parent1_repr, str)):
        raise ValueError("Parent 1 representation must be a list or a string")
    if not (isinstance(parent2_repr, list) or isinstance(parent2_repr, str)):
        raise ValueError("Parent 1 representation must be a list or a string")
    if len(parent1_repr) != len(parent2_repr):
        raise ValueError("Parent 1 and Parent 2 representations must be the same length")

    # Choose random crossover point
    xo_point = random.randint(1, len(parent1_repr) - 1)

    offspring1_repr = parent1_repr[:xo_point] + parent2_repr[xo_point:]
    offspring2_repr = parent2_repr[:xo_point] + parent1_repr[xo_point:]

    return offspring1_repr, offspring2_repr

def cycle_crossover(parent1_repr: str | list, parent2_repr: str | list):
    """
    Performs Cycle Crossover (CX) between two parents

    Cycle Crossover preserves the position of elements by identifying a cycle
    of indices where the values from each parent will be inherited by each offspring.
    The remaining indices are filled with values from the other parent, maintaining valid permutations.

    Args:
        parent1_repr (str or list): The first parent representation.
        parent2_repr (str or list): The second parent representation.
            Both parents must have the same length and type.

    Returns:
        tuple: Two offspring permutations resulting from the crossover.
    """
    # Randomly choose a starting index for the cycle
    initial_random_idx = random.randint(0, len(parent1_repr)-1)

    # Initialize the cycle with the starting index
    cycle_idxs = [initial_random_idx]
    current_cycle_idx = initial_random_idx

    # Traverse the cycle by following the mapping from parent2 to parent1
    while True:
        value_parent2 = parent2_repr[current_cycle_idx]
        # Find where this value is in parent1 to get the next index in the cycle
        next_cycle_idx = parent1_repr.index(value_parent2)

        # Closed the cycle -> Break
        if next_cycle_idx == initial_random_idx:
            break

        cycle_idxs.append(next_cycle_idx)
        current_cycle_idx = next_cycle_idx
    
    offspring1_repr = []
    offspring2_repr = []
    for idx in range(len(parent1_repr)):
        if idx in cycle_idxs:
            # Keep values from parent1 in offspring1 in the cycle indexes
            offspring1_repr.append(parent1_repr[idx])
            # Keep values from parent2 in offspring2 in the cycle indexes
            offspring2_repr.append(parent2_repr[idx])
        else:
            # Swap elements from parents in non-cycle indexes
            offspring1_repr.append(parent2_repr[idx])
            offspring2_repr.append(parent1_repr[idx])

    # To keep the same type as the parents representation
    if isinstance(parent1_repr, str) and isinstance(parent2_repr, str):
        offspring1_repr = "".join(offspring1_repr)
        offspring2_repr = "".join(offspring2_repr)

    return offspring1_repr, offspring2_repr