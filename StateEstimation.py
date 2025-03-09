import numpy as np
from rotations import Quaternion, skew_symmetric

"""
Code Responsibility: This is the code for state estimation of the car using the
Error-State Extended Kalman Filter (EKF) algorithm.

Will use the Inertial Measurement Unit (IMU) data as continuous input to
estimate the current state and GPS or/and LiDAR measurements to correct the
state estimate.

The state estimate vector will contain : Xk = [Pk, Vk, Qk].T in R(10x1), where
        Pk = the 3D position
        Vk = the 3D velocity
        Qk = the 4D unit quaternion representing the orientation

    Note: All the coordinates ar in the navigational frame.

Motion model:
    input: u = [fk, wk].T in R(6x1), where
                fk = 3D specific force in the body frame
                wk = 3D angular rate in the body frame

    Equations:
        Pk = Pk-1 + Vk-1 * dt + 0.5 * dt^2 * (Cns * fk + g)
        Vk = Vk-1 + dt * (Cns * fk + g)
        Qk = Qk-1 * Q(wk * dt) -> quaternion multiplication

Linearized motion model:
    delta_xk = [delta_Pk, delta_Vk, delta_Qk].T in R(9x1)

    Fk = [[I3, I3 * dt, 0],
          [0, I3, -[Cns * fk]_x * dt],
          [0, 0, I3]] in R(9x9)

    L = Lk = [[O3, O3],
              [I3, O3],
              [O3, I3]] in R(9x6)

Measurement noise distribution:
    R = delta_t ** 2 * [[var_imu_f ** 2
                                        var_imu_w ** 2]] in R(6x6)

Algorithm:
    1. Update state w/ IMU inputs:
        Xk_predicted = ...
    2. Propagate uncertainty:
        Pk_predicted = ...
    If GPS/LiDAR measurement available:
        3. Compute Kalman gain:
            Kk = ...
        4. Correct error-state estimate:
            delta_xk = ...
        5. Correct preddicted state:
            Xk = ...
        6. Compute corrected covariance:
            Pk = ...

    Back to step 1.
    
"""

# Define measurment update with Kalman equations.
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain
    R = np.eye(3) @ sensor_var
    K = p_cov_check @ h_jac.T @ np.linalg.inv(h_jac @ p_cov_check @ h_jac.T + R)

    # 3.2 Compute error state
    error_state = K @ (y_k - p_check)

    # 3.3 Correct predicted state
    p_hat = p_check + error_state[0:3]
    v_hat = v_check + error_state[3:6]
    q_hat = Quaternion(euler=error_state[6:9]).quat_mult_left(q_check)

    # 3.4 Compute corrected covariance
    p_cov_hat = (np.eye(9) - K @ h_jac) @ p_cov_check

    return p_hat, v_hat, q_hat, p_cov_hat

# Set the variances of the sensors.
var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.01
var_lidar = 0.5

# Constants and precomputed jacobians.
g = np.array([0, 0, -9.81]) # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)    # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)    # measurement model jacobian

# State vectors.
p_est = np.zeros([imu_f.data.shape[0], 3])      # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])      # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])      # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])   # covariance matrices at each timestep

# Set initial values.
p_est[0] = ...
v_est[0] = ...
q_est[0] = Quaternion(euler=...).to_numpy()
p_cov[0] = np.eye(9)  # covariance of estimate
gnss_i  = 0
lidar_i = 0

while True:
    # Receive data from IMU
    imu_f_data = ...
    imut_w_data = ...
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update state with IMU inputs
    C_ns = Quaternion(*q_est[k - 1]).to_mat()
    C_ns_dot_fk = C_ns @ imu_f.data[k - 1]
    p_est[k] = p_est[k - 1] + delta_t * v_est[k - 1] + 0.5 * delta_t**2 * (C_ns_dot_fk + g)
    v_est[k] = v_est[k - 1] + delta_t * (C_ns_dot_fk + g)
    q_est[k] = Quaternion(axis_angle=imu_w.data[k - 1] * delta_t).quat_mult_right(q_est[k - 1])

    # 1.1 Linearize the motion model and compute Jacobians
    F = np.eye(9)
    F[:3, 3:6] = delta_t * np.eye(3)
    F[3:6, 6:9] = -skew_symmetric(C_ns_dot_fk) * delta_t
    Q = np.eye(6)
    Q[:3, :3] *= var_imu_f**2 * delta_t**2
    Q[3:, 3:] *= var_imu_w**2 * delta_t**2

    # 2. Propagate uncertainty
    p_cov[k] = F @ p_cov[k - 1] @ F.T + l_jac @ Q @ l_jac.T

    # 3. Check availability of GNSS and LIDAR measurements
    if lidar_i < lidar.data.shape[0] and lidar.t[lidar_i] == imu_f.t[k - 1]:
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_lidar * np.eye(3), p_cov[k], lidar.data[lidar_i], p_est[k], v_est[k], q_est[k])
        lidar_i += 1

    if gnss_i < gnss.data.shape[0] and gnss.t[gnss_i] == imu_f.t[k - 1]:
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_gnss * np.eye(3), p_cov[k], gnss.data[gnss_i], p_est[k], v_est[k], q_est[k])
        gnss_i += 1

    # Send the state-estimation to ?


"""
TODO: 1) Define the data structure where to keep the IMU/GNSS/LiDAR data.
      2) Define the communication between the output sensor and this.
      3) Test it.

Questions:
    1) Keep all the data or just the prev step ?
    2) How should our structure be sent ? {data, timestamp}

"""