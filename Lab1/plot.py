import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

#Equation Task 3
# p(t)=∫∫a(t)dtdt+v0t+p0

#Equation Task 4
# θ(t)=∫ω(t)dt+θ0

def remove_outliers_iqr(data, factor=3):
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
    return mask

def filter(data_to_filter, outliers_filter_factor, window_size):
    # Medium filter
    data_filtered = np.apply_along_axis(medfilt, 0, data_to_filter, kernel_size=window_size)

    if outliers_filter_factor == -1:
        return (time, data_filtered) 
    # Remove outliers
    mask = remove_outliers_iqr(data_filtered, outliers_filter_factor)
    data_cleaned = data_filtered[mask]
    time_cleaned = time[mask]
    return time_cleaned, data_cleaned

def reconstruct_trajectory(time, accel_data):
    # Remove gravity
    gravity = np.array([0, 0, 981])
    accel_without_gravity = accel_data - gravity

    # Calculate time differences
    dt = np.diff(time)

    # Integrate acceleration to get velocity
    velocity = np.cumsum(accel_data[:-1] * dt[:, np.newaxis], axis=0) / 1000
    velocity = np.vstack([[0, 0, 0], velocity])  # Initial velocity

    # Integrate velocity to get position
    position = np.cumsum(velocity[:-1] * dt[:, np.newaxis], axis=0)
    position = np.vstack([[0, 0, 0], position])  # Initial position

    return position

def reconstruct_orientation(time, angular_velocity):
    # Calculate time differences
    dt = np.diff(time)
    
    # Integrate angular velocity to get orientation
    orientation = np.cumsum(angular_velocity[:-1] * dt[:, np.newaxis], axis=0)
    orientation = np.vstack([[0, 0, 0], orientation])  # Initial orientation
    
    return orientation

#Load Data (Team 15)
data = np.loadtxt('Lab1/LAB1_15.txt')

#Split Data
time = data[:, 0] / 1000000
linear_velocity = data[:, 1:4]
angular_velocity = data[:, 4:7] 

# Filter out low values for the gyro
angular_velocity[np.abs(angular_velocity) <= 5] = 0

# Filter
time_linear_velocity, linear_velocity_filtered = filter(linear_velocity, 20, 15)
time_angular_velocity, angular_velocity_filtered = filter(angular_velocity, -1, 9)

# Reconstruction
trajectory = reconstruct_trajectory(time_linear_velocity, linear_velocity_filtered)
orientation = reconstruct_orientation(time_angular_velocity, angular_velocity_filtered)

########################################## Create the plot ############################################
fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.plot(time, linear_velocity[:, 0], label='X')
ax1.plot(time, linear_velocity[:, 1], label='Y')
ax1.plot(time, linear_velocity[:, 2], label='Z')
ax1.set_ylabel('Linear Velocity')
ax1.legend()

ax1_ang = ax1.twinx()
ax1_ang.plot(time, angular_velocity[:, 0], label='RX', linestyle='--')
ax1_ang.plot(time, angular_velocity[:, 1], label='RY', linestyle='--')
ax1_ang.plot(time, angular_velocity[:, 2], label='RZ', linestyle='--')
ax1_ang.set_ylabel('Angular Velocity')
ax1_ang.legend()
ax1.set_title('Unfiltered')
ax1.set_xlabel('Time [s]')
ax1.grid(True)

# Set symmetrical limits to align 0
y_max_angular = max(abs(np.min(angular_velocity)), abs(np.max(angular_velocity)))
ax1_ang.set_ylim(-y_max_angular, y_max_angular)

ax2 = fig.add_subplot(212)
ax2.plot(time_linear_velocity, linear_velocity_filtered[:, 0], label='X (filt)')
ax2.plot(time_linear_velocity, linear_velocity_filtered[:, 1], label='Y (filt)')
ax2.plot(time_linear_velocity, linear_velocity_filtered[:, 2], label='Z (filt)')
ax2.set_ylabel('Filtered Linear Velocity')
ax2.legend()

ax2_ang = ax2.twinx()
ax2_ang.plot(time_angular_velocity, angular_velocity_filtered[:, 0], label='RX (filt)', linestyle='--')
ax2_ang.plot(time_angular_velocity, angular_velocity_filtered[:, 1], label='RY (filt)', linestyle='--')
ax2_ang.plot(time_angular_velocity, angular_velocity_filtered[:, 2], label='RZ (filt)', linestyle='--')
ax2_ang.set_ylabel('Filtered Angular Velocity')
ax2_ang.legend()

ax2.set_title('Filtered')
ax2.set_xlabel('Time [s]')
ax2.grid(True)

plt.tight_layout(pad=0.5)

# Set symmetrical limits to align 0
ax2_ang.set_ylim(-y_max_angular, y_max_angular)

fig.align_ylabels([ax1, ax2])
fig.align_ylabels([ax1_ang, ax2_ang])

fig2 = plt.figure()
ax3 = fig2.add_subplot(212, projection='3d')
ax3.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
ax3.set_xlabel('X Position')
ax3.set_ylabel('Y Position')
ax3.set_zlabel('Z Position')
ax3.set_title('Reconstructed XYZ Trajectory')
ax3.legend()
ax3.grid(True)

ax4 = fig2.add_subplot(211, projection='3d')
ax4.plot(orientation[:, 0], orientation[:, 1], orientation[:, 2])
ax4.set_xlabel('Roll')
ax4.set_ylabel('Pitch')
ax4.set_zlabel('Yaw')
ax4.set_title('Reconstructed Orientation')
ax4.legend()
ax4.grid(True)

plt.show()