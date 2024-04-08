import numpy as np

### my lib

import utils
import images
import distances
import transforms


def rotation_matrix_2d(theta):
    
    return np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                     [np.sin(np.deg2rad(theta)),  np.cos(np.deg2rad(theta))]])


def rotation_matrix_to_angle(R):

    return np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))


def get_cumulative_rotation(rotations):
    
    return np.sum(rotations) % 360


def get_shift_grid_array(grid_size=1):
    """generate shift array for s x s"""
    
    ### simplify this function
    
    grid = []
    
    for s1 in range(-grid_size, grid_size+1):
        for s2 in range(-grid_size, grid_size+1):
            grid.append([s1, s2])
        
    return np.array(grid)


def get_rotation_alignment_angles_from_distances(distance_dict, angles):
    
    rotation_angles = {}
    
    for idx, dists in distance_dict.items():
        rotation_angles[idx] = angles[np.argmin(dists)]
        
    return rotation_angles
        
        
def rotational_alignment_image_stack_to_reference(images, distance_dict, angles):
    
    images_aligned = np.zeros(images.shape)
    rotation_angles = get_rotation_alignment_angles_from_distances(distance_dict, angles)
    
    for idx, image in enumerate(images):
        images_aligned[idx] = utils.rotate(image, -rotation_angles[idx])
        
    return images_aligned


def translation_image_alignment(image1, image2):
    
    ty, tx = distances.fast_translation_distance(image1, image2)
    image_2_align = utils.translate(image2, -ty, -tx)
    
    return image2_align


def reference_translation_distances(reference, images):
    ### is there a way to avoid calling fft multiple times when combined with other functions
    ###  - e.g. just using phase shift to update?
    
    shift_dict = {}
    ref_ft = np.conj(np.fft.fft2(reference))
    
    for idx, image in enumerate(images):
        image_ft = np.fft.fft2(image)
        corr = np.fft.ifft2(ref_ft * image_ft)
        ty, tx = np.unravel_index(np.argmax(corr), corr.shape)
        shift_dict[idx] = [ty, tx]
        
    return shift_dict


def translation_alignment_image_stack_to_reference(images, shift_dict):
    
    images_aligned = np.zeros(images.shape)
        
    for idx, image in enumerate(images):
        ty, tx = shift_dict[idx]
        images_aligned[idx] = utils.translate(image, -ty, -tx)
        
    return images_aligned


def iterative_rotation_translation_alignment_image_stack_to_reference(reference, images, metric='sw', n_iterations=3):
    ### input is preprocessed images
    ### decide whether rotation or translation first
    ### log to keep track of R,T for comparison to gt
    ### option to have log keep distance matrix (or dict)
    ### need to specify which rotation alignment to use (make separate functions?)
    ### need to fix reference input containing a duplicate
    ### doesn't really make sense to have L2 translation as last step
    ### also add n_points and angles as optional inputs
    ### should probably remask images after iteration to bound
    
    N, ny, nx = images.shape
    n_points = ny
    angles = np.linspace(0, 360, ny, endpoint=False)
    
    log = {idx: [np.zeros(n_iterations), np.zeros((n_iterations, 2))] for idx in range(N)}
    
    ref_pos, ref_neg = transforms.Transform(reference).signed_inverse_cdf_transform()
    
    for n in range(n_iterations):
        
        ### rotation alignment
        if metric == 'sw':
            imgs_pos, imgs_neg = transforms.Transform(images).signed_inverse_cdf_transform()
            dists_dict = distances.reference_rotational_distances(ref_pos[0], imgs_pos, n_points)  # fix the ref[0] part
            rotation_angles = get_rotation_alignment_angles_from_distances(dists_dict, angles)
            
        elif metric == 'bf':
            dists_dict = {idx: distances.real_space_rotational_distance(reference[0], images[idx], angles) for idx in range(N)}
            rotation_angles = get_rotation_alignment_angles_from_distances(dists_dict, angles)
        
        ### translation alignement
        images = rotational_alignment_image_stack_to_reference(images, dists_dict, angles)
        shift_dict = reference_translation_distances(reference[0], images)
        images = translation_alignment_image_stack_to_reference(images, shift_dict)
        
        ### update log
        for idx in range(N):
            log[idx][0][n] = rotation_angles[idx]
            log[idx][1][n] = shift_dict[idx]
        
    return images, log


def shift_grid_rotation_alignment_image_stack_to_reference(reference, images, grid_size=1):
    ### similar to iterative - has issue that reference is duplication image in stack
    ### need to simplify this function by splitting into parts
    ### possibly incorporate fourier tricks to avoid recomputing the Transforms
    ### include options to pass angles and n_points as input
    ### note that utils translate does circular shift
    ### add option for different metrics
    ### replace images_aligned lines with functions from alignments
    ### should probably remask images after iteration to bound
    
        
    N, ny, nx = images.shape
    n_points = ny
    angles = np.linspace(0, 360, ny, endpoint=False)    
    
    shift_grid = get_shift_grid_array(grid_size)
    images_shift = np.zeros(images.shape)
    dists_log = {idx: [] for idx in range(N)}
    
    ref_pos, ref_neg = transforms.Transform(reference).signed_inverse_cdf_transform()
        
    for shift in shift_grid:
        ty, tx = shift
        
        for idx in range(N):
            images_shift[idx] = utils.translate(images[idx], ty, tx)
            
        imgs_pos, imgs_neg = transforms.Transform(images_shift).signed_inverse_cdf_transform()
        dists_dict = distances.reference_rotational_distances(ref_pos[0], imgs_pos, n_points)
        
        for idx, dist in dists_dict.items():
            d_min_idx = np.argmin(dist)
            d_min = dist[d_min_idx]
            a_min = angles[d_min_idx]
            dists_log[idx].append((shift, a_min, d_min))

    shift_dict = {}
    rot_dict = {}
    
    for idx, values in dists_log.items():
        dist_mins = np.array([d[2] for d in values])
        dist_min_idx = np.argmin(dist_mins)
        shift_min = shift_grid[dist_min_idx]
        a_min = values[dist_min_idx][1]
        
        shift_dict[idx] = shift_min
        rot_dict[idx] = np.array(a_min)
        
    ### recover aligned images (check order of operations here T @ R @ A)
    images_aligned = np.array([utils.translate(images[idx], ty, tx) for idx, (ty, tx) in shift_dict.items()])
    images_aligned = np.array([utils.rotate(images_aligned[idx], -a) for idx, a in rot_dict.items()])
    
    log = {idx: [rot_dict[idx], shift_dict[idx]] for idx in range(N)}
    
    return images_aligned, log
