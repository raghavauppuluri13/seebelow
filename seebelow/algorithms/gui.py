import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from seebelow.utils.constants import HISTORY_DTYPE


class HeatmapAnimation:

    def __init__(self, data, ground_truth=None, cmap="hot", interval=200):
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

        grid_size = data[0]["grid"].shape
        assert data.dtype == HISTORY_DTYPE(grid_size), print(data.dtype, "!=",
                                                             HISTORY_DTYPE(grid_size))
        self.ground_truth = ground_truth

        # Assuming each frame in data contains two tuples for two heatmaps
        self.nx, self.ny = grid_size

        plots_cnt = 2 if ground_truth is not None else 1

        if self.ground_truth is not None:
            assert ground_truth.shape == grid_size
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, plots_cnt, figsize=(10, 5))
            self.heatmap2 = self.ax2.imshow(
                np.zeros((self.nx, self.ny)),
                cmap=self.cmap,
                interpolation="nearest",
                aspect="equal",
            )
            self.fig.colorbar(self.heatmap2, ax=self.ax2)
        else:
            self.fig, self.ax1 = plt.subplots(1, plots_cnt, figsize=(10, 5))

        self.heatmap1 = self.ax1.imshow(
            np.zeros((self.nx, self.ny)),
            cmap=self.cmap,
            interpolation="nearest",
            aspect="equal",
        )

        # Add color bar for the heatmaps
        self.fig.colorbar(self.heatmap1, ax=self.ax1)
        self.frame_label = self.ax1.text(0.02,
                                         0.95,
                                         '',
                                         transform=self.ax1.transAxes,
                                         fontsize=12,
                                         verticalalignment='top',
                                         color='white')

        # Initialize the markers for the 'X' marks
        init_mark1 = data[0]["sample_pt"]
        (self.x_mark1, ) = self.ax1.plot(init_mark1[1], init_mark1[0], "wx", markersize=10)

        # Create the animation object
        self.ani = FuncAnimation(self.fig, self.update, frames=self.frames, interval=self.interval)

    def update(self, frame):
        """Update function for animation."""
        data_frame1 = self.data[frame]["grid"]
        next_mark1 = self.data[frame]["sample_pt"]

        # Update the data for both heatmaps
        self.heatmap1.set_data(data_frame1)

        # Update the positions of the 'X' marks
        self.x_mark1.set_data(next_mark1[1], next_mark1[0])

        self.frame_label.set_text(f'Frame: {frame}')

        # Update color limits for both heatmaps
        self.heatmap1.set_clim(vmin=np.min(data_frame1), vmax=np.max(data_frame1))

        if self.ground_truth is not None:
            self.heatmap2.set_clim(vmin=np.min(self.ground_truth), vmax=np.max(self.ground_truth))
            self.heatmap2.set_data(self.ground_truth)
            return self.heatmap1, self.x_mark1, self.heatmap2

        return self.heatmap1, self.x_mark1

    def visualize(self):
        """Visualize the animation."""
        plt.show()

    def save_animation(self, filename):
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
    data = [(((np.random.randint(nx), np.random.randint(ny)), np.random.rand(nx, ny)), )
            for _ in range(frames)]

    gt = (np.random.randint(nx), np.random.randint(ny)), np.random.rand(nx, ny)

    animation = HeatmapAnimation(data, gt)
    animation.save_animation("gaussian_process.mp4")
