import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib

# Ensure that the FFmpeg writer is available
matplotlib.use("Agg")


class HeatmapAnimation:
    def __init__(self, data, cmap="hot", interval=200):
        """
        data: BxNxM numpy array, where B is the number of frames and NxM is the grid size.
        cmap: Colormap for the heatmap.
        interval: Time interval between frames in milliseconds.
        x_mark_pos: Tuple of (row, col) to place the 'X' mark.
        """
        self.data = data
        self.cmap = cmap
        self.interval = interval
        self.frames = len(data)
        self.nx, self.ny = data[0][1].shape
        self.fig, self.ax = plt.subplots()
        self.heatmap = self.ax.imshow(
            np.zeros((self.nx, self.ny)),
            cmap=self.cmap,
            interpolation="nearest",
            aspect="auto",
        )
        init_mark = data[0][0]
        (self.x_mark,) = self.ax.plot(
            init_mark[1], init_mark[0], "wx", markersize=10
        )  # Initialize the marker for the X

    def update(self, frame):
        """Update function for animation."""
        next_mark, data_frame = self.data[frame]
        self.heatmap.set_data(data_frame)
        self.heatmap.set_clim(vmin=np.min(data_frame), vmax=np.max(data_frame))
        # Update the position of the 'X' mark
        self.x_mark.set_data(next_mark[1], next_mark[0])
        return self.heatmap, self.x_mark

    def add_gridlines(self):
        """Add gridlines to the plot."""
        self.ax.set_xticks(np.arange(-0.5, self.ny, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.nx, 1), minor=True)
        self.ax.grid(which="minor", color="black", linestyle="-", linewidth=2)
        self.ax.tick_params(which="minor", size=0)

    def save_animation(self, filename="heatmap_animation.mp4"):
        """Save the animation as an MP4 file."""
        print("Saving animation to", filename)
        ani = FuncAnimation(
            self.fig, self.update, frames=self.frames, interval=self.interval, blit=True
        )
        Writer = matplotlib.animation.writers["ffmpeg"]
        writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
        ani.save(filename, writer=writer)


# Example usage:
if __name__ == "__main__":
    data = np.random.rand(10, 100, 100)  # Example data, replace with your actual data
    x_mark_position = (50, 50)  # Replace with the actual (row, col) for the 'X' mark
    animation = HeatmapAnimation(data)
    animation.add_gridlines()
    animation.save_animation("gaussian_process.mp4")
