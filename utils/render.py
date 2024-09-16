import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter


class MultiSampleVideoRenderer:
    
    def __init__(self, data, pad1_data, pad2_data):
        self.videos = []
        for i in range(len(data)):
            self.videos.append(VideoRenderer(data[i], pad1_data[i], pad2_data[i]))
            
    def render(self):
        render(self.videos, fps=self.videos[0].fps)
        
class VideoRenderer:
    
    def __init__(self, data, pad1_data, pad2_data):
        self.load(data, pad1_data, pad2_data)
        
    def load(self, data, pad1_data, pad2_data):        
        # Load metadata
        self.fps = data[0, 0, 0]
        self.num_frames = int(data[0, 0, 1])
        self.num_frames_usable = int(data[0, 0, 2])
        
        # Load player and ball data
        self.player1 = data[:, 2:27, :]  # Assuming 44 keypoints for each player
        self.player2 = data[:, 27:52, :]
        self.ball    = data[:, 52, :]
        self.pad1_data = pad1_data
        self.pad2_data = pad2_data
        
        # Replace NaN values with None for ball positions
        self.ball = [b if not np.isnan(b).any() else None for b in self.ball]
        
    def __getitem__(self, i):
        return self.player1[i], self.player2[i], self.ball[i], self.pad1_data[i], self.pad2_data[i]
        
    def __len__(self):
        return self.num_frames_usable

def quaternion_to_rotation_matrix(q):
    """Convert a quaternion to a rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
def render_paddle(ax, pose, radius, color, rescale_factor):
    center = pose[:3] * rescale_factor
    quat = pose[3:]
    
    # Create a circle of points
    theta = np.linspace(0, 2*np.pi, 20)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)
    
    # Create a line from the center to the edge of the paddle
    line1_x = np.array([0, 2*radius])
    line1_y = np.zeros(2)
    line1_z = np.zeros(2)
    line2_x = np.zeros(2)
    line2_y = np.zeros(2)
    line2_z = np.array([0, 2*radius])
    
    # Combine points into a single array
    circle_points = np.column_stack((x, y, z))
    line1_points = np.column_stack((line1_x, line1_y, line1_z))
    line2_points = np.column_stack((line2_x, line2_y, line2_z))
    
    # Create rotation matrix from quaternion
    rotation_matrix = quaternion_to_rotation_matrix(quat)
    
    # Apply rotation to points
    rotated_circle = np.dot(circle_points, rotation_matrix.T)
    rotated_line1 = np.dot(line1_points, rotation_matrix.T)
    rotated_line2 = np.dot(line2_points, rotation_matrix.T)
    
    # Translate points to center position
    translated_circle = rotated_circle + center
    translated_line1 = rotated_line1 + center
    translated_line2 = rotated_line2 + center
    
    # Create the paddle circle
    paddle = ax.plot(translated_circle[:, 0], translated_circle[:, 1], translated_circle[:, 2], color=color, alpha=0.7)[0]
    
    # Create the orientation line
    orientation_line1 = ax.plot(translated_line1[:, 0], translated_line1[:, 1], translated_line1[:, 2], color='black', linewidth=1)[0]
    orientation_line2 = ax.plot(translated_line2[:, 0], translated_line2[:, 1], translated_line2[:, 2], color='green', linewidth=1)[0]
    
    # Fill the paddle
    verts = list(map(tuple, translated_circle))
    paddle_fill = art3d.Poly3DCollection([verts], alpha=0.3)
    paddle_fill.set_color(color)
    ax.add_collection3d(paddle_fill)
    
    return paddle, paddle_fill, orientation_line1, orientation_line2

def render(processed_videos, fps, paddle_radius=0.08, show_feet=False, show_extended=False):
    RESCALE_FACTOR = 0.003048
    
    min_frame, max_frame = 0, processed_videos[0].num_frames if show_extended else len(processed_videos[0])
    scenes = [ { j: processed_videos[i][j] for j in range(max_frame) } for i in range(len(processed_videos)) ]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the vertices of the table
    table_dims = [900 * RESCALE_FACTOR, 500 * RESCALE_FACTOR, 250 * RESCALE_FACTOR]
    
    # Add the ping pong table to the scene
    def add_table(ax):
        table_x = [-table_dims[0] / 2, table_dims[0] / 2, table_dims[0] / 2, -table_dims[0] / 2, -table_dims[0] / 2]
        table_y = [-table_dims[1] / 2, -table_dims[1] / 2, table_dims[1] / 2, table_dims[1] / 2, -table_dims[1] / 2]
        table_z = [table_dims[2]] * 5
        
        # Draw the table outline
        ax.plot(table_x, table_y, table_z, color='b')
        
        # Add a colored surface for the table top
        X, Y = np.meshgrid([-table_dims[0] / 2, table_dims[0] / 2], [-table_dims[1] / 2, table_dims[1] / 2])
        Z = np.full_like(X, table_dims[2])
        ax.plot_surface(X, Y, Z, color='darkblue', alpha=0.5)

    bounds = 1300 * RESCALE_FACTOR
    ball_trajectories = [[] for _ in range(len(scenes))]

    def update(frame):            
        ax.cla()  # Clear the current axes
        
        for i in range(len(scenes)):
            scene = scenes[i]
            ball_trajectory = ball_trajectories[i]
            
            if frame in scene:
                ax.set_xlim(-bounds, bounds)
                ax.set_ylim(-bounds, bounds)
                ax.set_zlim(0, bounds)
                
                p1_keypoints, p2_keypoints, ball_pos, pad1_pose, pad2_pose = scene[frame]
                
                # Plot the players.
                points = np.concatenate((p1_keypoints, p2_keypoints), axis=0) * RESCALE_FACTOR
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.5, color="black" if i == 0 else "blue")
                
                # Render paddles
                render_paddle(ax, pad1_pose, paddle_radius, "red", RESCALE_FACTOR)
                render_paddle(ax, pad2_pose, paddle_radius, "red", RESCALE_FACTOR)
                if ball_pos is not None:
                    ax.scatter(ball_pos[0] * RESCALE_FACTOR, ball_pos[1] * RESCALE_FACTOR, ball_pos[2] * RESCALE_FACTOR, s=4, color="green")
                if frame == 0:
                    ball_trajectory.clear()
                if ball_pos is not None:
                    ball_trajectory.append(ball_pos)
                    # Keep only the last 3 positions
                    if len(ball_trajectory) > 5:
                        ball_trajectory.pop(0)
                    ball_trajectory_arr = np.array(ball_trajectory)
                    ax.plot(ball_trajectory_arr[:, 0] * RESCALE_FACTOR, ball_trajectory_arr[:, 1] * RESCALE_FACTOR, ball_trajectory_arr[:, 2] * RESCALE_FACTOR, color="red")
                    
        if frame in scenes[0]:    
            add_table(ax)
                
            # Add title and labels to the axes
            ax.set_title('3D Reconstruction of Ping Pong Rally')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_zlabel('Z (meters)')
            
        return ax,
    
    ani = FuncAnimation(fig, update, frames=range(min_frame, max_frame+1), interval=1000/fps, repeat_delay=2000, repeat=True)  # interval in milliseconds
    ani.save("rec.gif", writer='pillow')
    plt.show()
