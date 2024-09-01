import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from neurodynex3.hopfield_network import plot_tools

def get_patterns(dim, num):
    cb_pattern = np.array([[(-1) ** (row + col) for col in range(dim)] for row in range(dim)])
    patterns = [cb_pattern]
    for _ in range(num - 1):
        rand_pattern = np.random.choice([-1, 1], (dim, dim), p=[0.5, 0.5])
        patterns.append(rand_pattern)
    return patterns

def perturb_pattern(original_pattern, num_flips):
    altered_pattern = original_pattern.copy()
    all_indices = list(np.ndindex(altered_pattern.shape))
    np.random.shuffle(all_indices)
    for index in all_indices[:num_flips]:
        altered_pattern[index] *= -1
    return altered_pattern

def calculate_weights(stored_patterns, dim):
    neuron_count = dim ** 2
    weight_matrix = np.zeros((neuron_count, neuron_count))
    for each_pattern in stored_patterns:
        flat_pattern = each_pattern.flatten()
        weight_matrix += np.outer(flat_pattern, flat_pattern)
    weight_matrix /= neuron_count
    np.fill_diagonal(weight_matrix, 0)
    return weight_matrix

def network_evolution(current_state, weight_matrix):
    state_vector = current_state.flatten()
    evolved_vector = np.sign(weight_matrix @ state_vector)
    evolved_vector[evolved_vector == 0] = 1
    return evolved_vector.reshape(current_state.shape)

grid_size = 4
total_patterns = 5
stored_patterns = get_patterns(grid_size, total_patterns)

plot_tools.plot_pattern_list(stored_patterns)

initial_cue = deepcopy(stored_patterns)
perturbed_state = perturb_pattern(initial_cue[0], 3)  # Perturb the first pattern by flipping 3 pixels

weights = calculate_weights(stored_patterns, grid_size)
states_sequence = [perturbed_state]

for _ in range(3):
    next_state = network_evolution(states_sequence[-1], weights)
    states_sequence.append(next_state)

plot_tools.plot_pattern(states_sequence[0])
plot_tools.plot_state_sequence_and_overlap(states_sequence, stored_patterns, reference_idx=0, suptitle="Network dynamics")
