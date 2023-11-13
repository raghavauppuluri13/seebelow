import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib


class HeatmapAnimation:
    def __init__(self, data, cmap="hot", interval=200):
        """
        data: List of tuples, where each tuple contains two tuples each with
              an 'X' mark position and a 2D numpy array for the heatmap data.
        cmap: Colormap for the heatmap.
        interval: Time interval between frames in milliseconds.
        """
        self.data = data
        self.cmap = cmap
        self.interval = interval
        self.frames = len(data)

        # Assuming each frame in data contains two tuples for two heatmaps
        self.nx, self.ny = data[0][1].shape

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.heatmap1 = self.ax1.imshow(
            np.zeros((self.nx, self.ny)),
            cmap=self.cmap,
            interpolation="nearest",
            aspect="equal",
        )
        self.heatmap2 = self.ax2.imshow(
            np.zeros((self.nx, self.ny)),
            cmap=self.cmap,
            interpolation="nearest",
            aspect="equal",
        )

        # Add color bar for the heatmaps
        self.fig.colorbar(self.heatmap1, ax=self.ax1)
        self.fig.colorbar(self.heatmap2, ax=self.ax2)

        # Initialize the markers for the 'X' marks
        init_mark1 = data[0][0]
        (self.x_mark1,) = self.ax1.plot(
            init_mark1[1], init_mark1[0], "wx", markersize=10
        )

        # Create the animation object
        self.ani = FuncAnimation(
            self.fig, self.update, frames=self.frames, interval=self.interval
        )

    def update(self, frame):
        """Update function for animation."""
        next_mark1, data_frame1, data_frame2 = self.data[frame]

        # Update the data for both heatmaps
        self.heatmap1.set_data(data_frame1)
        self.heatmap2.set_data(data_frame2)

        # Update the positions of the 'X' marks
        self.x_mark1.set_data(next_mark1[1], next_mark1[0])

        # Update color limits for both heatmaps
        self.heatmap1.set_clim(vmin=np.min(data_frame1), vmax=np.max(data_frame1))
        self.heatmap2.set_clim(vmin=np.min(data_frame2), vmax=np.max(data_frame2))

        return self.heatmap1, self.x_mark1, self.heatmap2

    def visualize(self):
        """Visualize the animation."""
        plt.show()

    def save_animation(self, filename="heatmap_animation.mp4"):
        """Save the animation as an MP4 file."""
        print("Saving animation to", filename)
        matplotlib.use("Agg")
        Writer = matplotlib.animation.writers["ffmpeg"]
        writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
        self.ani.save(filename, writer=writer)


# Example usage:
if __name__ == "__main__":
    frames = 10
    nx, ny = 100, 100
    # Each element in data is a tuple of tuples for two heatmaps and their 'X' marks
    data = [
        (
            ((np.random.randint(nx), np.random.randint(ny)), np.random.rand(nx, ny)),
            ((np.random.randint(nx), np.random.randint(ny)), np.random.rand(nx, ny)),
        )
        for _ in range(frames)
    ]

    animation = HeatmapAnimation(data)
    animation.save_animation("gaussian_process.mp4")
