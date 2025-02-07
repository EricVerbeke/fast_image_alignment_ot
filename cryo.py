import mrcfile
import numpy as np


def get_random_euler_angles(n=1):
    
    return np.random.uniform(0, 2*np.pi, (n, 3))


def get_random_euler_in_heimisphere(n=1):
    # use for a picking rotations in hemisphere relative to ref with R = I?
    
    euler_angles = np.zeros((n, 3))
    
    for idx in range(n):
        
        Z1 = np.random.uniform(0, 2*np.pi)
        Y1 = np.random.uniform(0, np.pi/2)
        Z2 = np.random.uniform(0, 2*np.pi)
        
        euler_angles[idx] = [Z1, Y1, Z2]
        
    return euler_angles


def get_rotation_axis_angular_difference_matrix(rotation_matrices):
    
    N = rotation_matrices.shape[0]
    
    rotation_axis_angular_difference = np.zeros((N,N))

    for i in range(N):
        R1 = rotation_matrices[i]
        for j in range(i+1, N):
            R2 = rotation_matrices[j]
            theta = angle_between_rotation_axis(R1, R2)

            rotation_axis_angular_difference[i, j] = theta
            rotation_axis_angular_difference[j, i] = theta

    return rotation_axis_angular_difference


def get_euler_from_gradient_over_Y(n, angle_max=np.pi, vary_Z1=False, vary_Z2=False):
    ### convention is for ZYZ INTRINSIC
    
    angles = np.zeros((n, 3))
    
    Y = np.linspace(0, angle_max, n, endpoint=True)
    # Y = np.linspace(0, angle_max, n+1, endpoint=True)[1:]  # if keep zero
    
    angles[:, 1] = Y
    
    if vary_Z1:
        angles[:, 0] = np.random.uniform(0, 2*np.pi, n)
    if vary_Z2:
        angles[:, 2] = np.random.uniform(0, 2*np.pi, n)
        
    return angles


def angle_between_rotation_axis(R1, R2):
    
    n1 = R1[:, 2]
    n2 = R2[:, 2]

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    theta = np.rad2deg(np.arccos(np.dot(n1, n2) / (n1_norm * n2_norm)))    
    
    return theta


def angle_distribution_from_nearest_neighbors_dict(knn, rotation_axis_matrix):
    
    angle_distribution = {}
    
    for k, neighbors in knn.items():
        angle_distribution[k] = rotation_axis_matrix[k, knn[k]]
        
    return angle_distribution


def angle_differences_from_nearest_neighbors(knn, rot_values):
    
    return [rot_values[k] for k in knn]


def b_factor_function(shape, voxel_size, B):
    """B factor equation as function of spatial frequency"""
    
    N = shape[0]
    
    spatial_frequency = np.fft.fftshift(np.fft.fftfreq(N, voxel_size))

    sf_grid = np.meshgrid(*[spatial_frequency**2 for dimension in range(len(shape))])

    square_sf_grid = sf_grid[0] # initialize to broadcast by dimension
    for dimension in range(1, len(shape)):
        square_sf_grid = square_sf_grid + sf_grid[dimension]
    
    G = np.exp(- square_sf_grid * (B/4))
    
    return G


def apply_b_factor(v, voxel, B_signal):
    """return array after applying B-factor decay, input is real array"""
    
    G = b_factor_function(v.shape, voxel, B_signal)
    V = np.fft.fftshift(np.fft.fftn(v))
    Vb = G * V
    vb = np.fft.ifftn(np.fft.ifftshift(Vb))
    
    return vb