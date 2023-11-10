import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Assuming a grid size and a function that generates the Gaussian process
grid_size = (100, 100)


def generate_gaussian_process(grid_size):
    # This is a placeholder for your Gaussian process generation logic
    # You would replace this with your actual computation of expectation and variance
    expectation = np.random.rand(*grid_size)
    variance = np.random.rand(*grid_size)
    return expectation, variance


# Initialize plot
fig, ax = plt.subplots()
heatmap = ax.imshow(np.zeros(grid_size), cmap="hot", interpolation="bilinear")


# Update function for animation
def update(frame):
    expectation, variance = generate_gaussian_process(grid_size)
    # The displayed heatmap could be based on expectation, variance, or a combination.
    # Here, we'll just show the expectation.
    heatmap.set_data(expectation)
    # Recompute color limits based on the new data for proper scaling
    heatmap.set_clim(expectation.min(), expectation.max())


# Create an animation
ani = FuncAnimation(fig, update, frames=100, interval=100)

# Show the animation
plt.show()
