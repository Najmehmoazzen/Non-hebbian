# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set coupling value
coupling_value = 16.0

# Define the custom cyclic colormap
colors = [
    (1.0, 0.0, 1.0),  # Magenta
    (1.0, 0.0, 0.0),  # Red
    (1.0, 1.0, 0.0),  # Yellow
    (0.0, 1.0, 0.0),  # Green
    (0.0, 1.0, 1.0),  # Cyan
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 0.0, 1.0)   # Magenta (loop back)
]
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

# Read and process the data, skipping the first column
phase_data = []
file_path = f"/home/Najmeh/Non-hebbian/simple_kuramoto/Save/Phases/k={coupling_value}00000.txt"
with open(file_path, "r") as textFile:
    for line in textFile:
        values = line.strip().split()
        if len(values) > 1:
            phase_data.append(values[1:])  # Skip the first column

# Convert to NumPy array and transpose
phase_data = np.array(phase_data, dtype=float).T  # Shape: (nodes, time)

# Wrap phases to [-π, π]
phase_data = (phase_data + np.pi) % (2 * np.pi) - np.pi

# Print info
print("Min phase (wrapped):", np.min(phase_data))
print("Max phase (wrapped):", np.max(phase_data))
print("Time steps:", phase_data.shape[1])

# Plotting
fig = plt.figure(figsize=(12, 6))

plt.imshow(phase_data, cmap=custom_cmap, aspect='auto', interpolation='nearest')
plt.colorbar(label='Phase (radians)')

# Axis settings
plt.xlim(0, phase_data.shape[1])
plt.gca().set_ylim(plt.gca().get_ylim()[::-1])  # Flip y-axis

plt.xlabel('Time', fontsize=12)
plt.ylabel('Node (i)', fontsize=12, labelpad=12)
plt.tick_params(axis='x', labelsize=12, labelcolor='#262626')
plt.tick_params(axis='y', labelsize=12, labelcolor='#262626')

# Save figure
plt.savefig(f'./pattern_k={coupling_value}.png', dpi=300, bbox_inches='tight')
plt.show()
