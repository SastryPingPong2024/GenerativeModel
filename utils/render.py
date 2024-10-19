import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.animation import FuncAnimation
import pickle

class MultiSampleVideoRenderer:
    
    def __init__(self, data, pad1_data, pad2_data, preds_start_index):
        self.videos = []
        self.preds_start_index = preds_start_index
        for i in range(len(data)):
            self.videos.append(VideoRenderer(data[i], pad1_data[i], pad2_data[i]))
        
        ball_positions = []
        for video in self.videos[1:]:
            ball_positions.append(video.ball)
        ball_positions = np.array(ball_positions)
        self.mean_ball_pos = ball_positions.mean(axis=0)
        self.std_ball_pos  = ball_positions.std(axis=0)
        
    def save(self):
        with open('rec.pkl', 'wb') as f:
            pickle.dump(self, f)
        
    def render(self):
        render(self.videos, fps=self.videos[0].fps, preds_start_index=self.preds_start_index, mean_ball_pos=self.mean_ball_pos, std_ball_pos=self.std_ball_pos)
    
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

def render(processed_videos, fps, paddle_radius=0.08, show_feet=False, show_extended=False, preds_start_index=None, mean_ball_pos=None, std_ball_pos=None):
    RESCALE_FACTOR = 0.003048

    # 80% coverage
    conformal_quantiles = np.array([
        [1.0360453161295955, 1.0060421730585092, 0.9345403759050802, 0.9006476186783563, 0.9270345399831427, 1.4393763350540143, 1.2094562937800228, 1.4135067796381076, 1.4623508452660876, 1.419807895759761, 1.3787020766732536, 1.3543019902531166, 1.325032998364788, 1.31463771518229, 1.3078959700825752, 1.2842044164824855, 1.1809334959638558, 1.1018509844331066, 1.0044412407633898, 0.9189904532055156, 0.8812098880899066, 0.8585306179284597, 0.8572847328852322, 0.9897980686118673, 1.1781973502991665, 1.36575077765098, 1.5594864097603427, 1.9073120415693643, 2.022261891610794, 2.5100727455804273, 2.934830111320364, 3.320472251721587, 3.0460921243493755, 2.8146241506640366], 
        [0.6900454405317427, 0.7473431578855383, 0.8094630353507111, 0.9191358642365861, 0.9903692957939794, 1.0990482325263065, 0.885583590359137, 0.9138004573502476, 0.9849818725453754, 1.0581199817629359, 1.1113109502150993, 1.1453538240426142, 1.1675913516569258, 1.1402480214124702, 1.1098206041635355, 1.048007726715955, 0.975562701735326, 0.9395120697681348, 0.8992127629430594, 0.8726612580858458, 0.8393749607945007, 0.8113302819994194, 0.8142066441481961, 0.8188848109484307, 0.826420229557527, 0.8264906200883683, 0.8224350583488859, 0.8752095561355656, 0.9600027349266504, 1.0003129307622753, 0.9540135768332403, 0.952657647411467, 1.270168116250955, 1.291647693266584], 
        [0.36517838918650647, 0.4854622984594174, 0.5619346156802857, 0.6527039680199961, 0.7281276025762704, 0.8382234466893768, 0.6212822043309962, 0.5446010764916672, 0.5195096892084868, 0.518887970015462, 0.5173677217317717, 0.5151165272794826, 0.47804486778014355, 0.43887818796395783, 0.41407155056730066, 0.4221283673774733, 0.4309543201741559, 0.40568442631265117, 0.38327306270727063, 0.39753494991867017, 0.4086800156578604, 0.4266356965448807, 0.4556734753078592, 0.47769293273595403, 0.5038886738860947, 0.5469222039879408, 0.6157063753115338, 0.7056187509559614, 0.8137272986372893, 0.844807245870312, 0.8124284047801136, 0.726928925053378, 0.8372933834895316, 1.4030637947292206] 
    ])

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
        table_z = [+table_dims[2]] * 5

        # Draw the table outline
        ax.plot(table_x, table_y, table_z, color='b')

        # Add a colored surface for the table top
        X, Y = np.meshgrid([-table_dims[0] / 2, table_dims[0] / 2], [-table_dims[1] / 2, table_dims[1] / 2])
        Z = np.full_like(X, table_dims[2])
        ax.plot_surface(X, Y, Z, color='darkblue', alpha=0.5)

        # Add table legs
        leg_positions = [
            (-table_dims[0] / 2, -table_dims[1] / 2),
            ( table_dims[0] / 2, -table_dims[1] / 2),
            ( table_dims[0] / 2,  table_dims[1] / 2),
            (-table_dims[0] / 2,  table_dims[1] / 2),
        ]

        for x, y in leg_positions:
            ax.plot([x, x], [y, y], [table_dims[2], 0], color='brown', linewidth=2)

    bounds = 3
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

                # Render paddles (optional, if data is available)
                # render_paddle(ax, pad1_pose, paddle_radius, "red", RESCALE_FACTOR)
                # render_paddle(ax, pad2_pose, paddle_radius, "red", RESCALE_FACTOR)

                if ball_pos is not None:
                    ball_pos_scaled = ball_pos * RESCALE_FACTOR
                    ax.scatter(ball_pos_scaled[0], ball_pos_scaled[1], ball_pos_scaled[2], s=4, color="black" if i == 0 else "green")

                    if i == 0 and preds_start_index and frame >= preds_start_index and frame <= 25 + preds_start_index:
                        t = round((frame - preds_start_index) * 30 / fps)
                        r = conformal_quantiles[:, t]
                        r = r * ((std_ball_pos[frame] * RESCALE_FACTOR) + 0.016 * (t + 1))

                        mean_ball_pos_scaled = mean_ball_pos[frame] * RESCALE_FACTOR
                        # Define the vertices of the box around the ball
                        box_x = [mean_ball_pos_scaled[0] - r[0], mean_ball_pos_scaled[0] + r[0]]
                        box_y = [mean_ball_pos_scaled[1] - r[1], mean_ball_pos_scaled[1] + r[1]]
                        box_z = [mean_ball_pos_scaled[2] - r[2], mean_ball_pos_scaled[2] + r[2]]

                        # Create the vertices for the bounding box
                        vertices = [
                            [box_x[0], box_y[0], box_z[0]],
                            [box_x[1], box_y[0], box_z[0]],
                            [box_x[1], box_y[1], box_z[0]],
                            [box_x[0], box_y[1], box_z[0]],
                            [box_x[0], box_y[0], box_z[1]],
                            [box_x[1], box_y[0], box_z[1]],
                            [box_x[1], box_y[1], box_z[1]],
                            [box_x[0], box_y[1], box_z[1]],
                        ]

                        # Define the faces of the bounding box
                        faces = [
                            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
                            [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
                            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Side faces
                            [vertices[1], vertices[2], vertices[6], vertices[5]],
                            [vertices[2], vertices[3], vertices[7], vertices[6]],
                            [vertices[3], vertices[0], vertices[4], vertices[7]],
                        ]

                        # Create the 3D polygon collection with darker edges
                        box = art3d.Poly3DCollection(
                            faces, alpha=0.2, facecolor='cyan', edgecolor='darkblue', linewidths=1.5
                        )
                        ax.add_collection3d(box)

                if frame == 0:
                    ball_trajectory.clear()
                if ball_pos is not None:
                    ball_trajectory.append(ball_pos)
                    # Keep only the last 5 positions
                    if len(ball_trajectory) > 5:
                        ball_trajectory.pop(0)
                    ball_trajectory_arr = np.array(ball_trajectory)
                    ax.plot(ball_trajectory_arr[:, 0] * RESCALE_FACTOR, ball_trajectory_arr[:, 1] * RESCALE_FACTOR, ball_trajectory_arr[:, 2] * RESCALE_FACTOR, color="black" if i == 0 else "red")

        if frame in scenes[0]:    
            add_table(ax)

            # Add title and labels to the axes
            ax.set_title('Ball Uncertainty Region Against Ground Truth Trajectory')
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_zlabel('Z (meters)')

        return ax,

    ani = FuncAnimation(fig, update, frames=range(min_frame, max_frame+1), interval=2.25*1000/fps, repeat_delay=2000, repeat=True)  # interval in milliseconds
    ani.save("rec.gif", writer='pillow')  # Saving is optional now
    plt.show()
