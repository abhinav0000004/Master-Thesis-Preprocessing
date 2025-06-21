from helper.util import group_intervals
import matplotlib.pyplot as plt

# helper for 2D and 3D plotting of c3d data

def plot_3d(self, x, y, z):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(self.x, self.y, self.z, c=range(self.framecount), marker='o')

    # Label the axes
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    ax.set_title(self.marker_names[self.keypoint_idx])

    # Show the plot
    plt.show()


def plot_2d(ax, title, x, y, z, missing_indices=[], corrupt_indices=[]):
    # Plot the points
    framecount = len(x)
    ax.plot(range(framecount), x, linewidth=2.0, label='x', color='tab:blue')
    ax.plot(range(framecount), y, linewidth=2.0, label='y', color='tab:orange')
    ax.plot(range(framecount), z, linewidth=2.0, label='z', color='tab:green')

    for p in group_intervals(missing_indices):
        ax.axvspan(p[0], p[1], color='#ff8080', alpha=0.2)
    for p in group_intervals(corrupt_indices):
        ax.axvspan(p[0], p[1], color='#ffed42', alpha=0.2)

    # Label the axes
    ax.set_xlabel('Frames')
    ax.set_ylabel('Deflection in mm')

    ax.set_title(title)
