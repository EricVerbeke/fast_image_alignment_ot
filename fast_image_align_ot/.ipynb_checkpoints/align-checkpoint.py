### standard libraries
import numpy as np

### my libraries
from fast_image_align_ot import images, transforms, distances, distances_ot, utils

### EV: TODO:
### - add interations as parameter for SD, CWD
### - add level as parameter for WEMD

def align_sw(reference, image):
    """
    compute rotational alignment in sliced 2-Wasserstein distance
    input: reference, image = two LxL arrays, pixel values range [0, +], float64
    output: alignment information and aligned image in starting dtype
    """
    
    ny, nx = reference.shape
    proj_angles = np.linspace(0, 360, ny, endpoint=False)
    n = ny + 1  # number of points to sample in NUFFT

    imgs = images.Image(np.array([reference, image])).preprocess_images()  # normalize the images
    imgs_transform = transforms.Transform(imgs, apply_ramp=False, angles=proj_angles, n_points=n).inverse_cdf_transform()
    U_I, V_I = imgs_transform[0], imgs_transform[1]

    dists = distances.rotational_distances(U_I, V_I)
    a_min = proj_angles[np.argmin(dists)]
    image_aligned = utils.rotate(image, -a_min)

    alignment = {'angles': proj_angles, 'distances': dists, 'min_theta': a_min, 'aligned_image': image_aligned}
    
    return alignment

def align_rfsw(reference, image):
    """
    compute rotational alignment in ramp-filtered sliced 2-Wasserstein distance
    input: reference, image = two LxL arrays, pixel values range [0, +], float64
    output: alignment information and aligned image in starting dtype
    """
    
    ny, nx = reference.shape
    proj_angles = np.linspace(0, 360, ny, endpoint=False)
    n = ny + 1  # number of points to sample in NUFFT

    imgs = images.Image(np.array([reference, image])).preprocess_images()  # normalize the images
    imgs_transform = transforms.Transform(imgs, apply_ramp=True, angles=proj_angles, n_points=n).signed_inverse_cdf_transform()
    t_pos, t_neg = imgs_transform[0], imgs_transform[1]
    U_Ip, U_In = t_pos[0], t_neg[0]
    V_Ip, V_In = t_pos[1], t_neg[1]

    dists = distances.signed_rotational_distances(U_Ip, V_Ip, U_In, V_In)
    a_min = proj_angles[np.argmin(dists)]
    image_aligned = utils.rotate(image, -a_min)

    alignment = {'angles': proj_angles, 'distances': dists, 'min_theta': a_min, 'aligned_image': image_aligned}
    
    return alignment


def align_msw(reference, image):
    """
    compute rotational alignment in max-sliced 2-Wasserstein distance
    input: reference, image = two LxL arrays, pixel values range [0, +], float64
    output: alignment information and aligned image in starting dtype
    """

    ny, nx = reference.shape
    proj_angles = np.linspace(0, 360, ny, endpoint=False)
    n = ny + 1  # number of points to sample in NUFFT

    imgs = images.Image(np.array([reference, image])).preprocess_images()  # normalize the images
    imgs_transform = transforms.Transform(imgs, apply_ramp=False, angles=proj_angles, n_points=n).inverse_cdf_transform()
    U_I, V_I = imgs_transform[0], imgs_transform[1]

    dists = distances.rotational_max_sliced_wasserstein(U_I, V_I, ny)
    a_min = proj_angles[np.argmin(dists)]
    image_aligned = utils.rotate(image, -a_min)

    alignment = {'angles': proj_angles, 'distances': dists, 'min_theta': a_min, 'aligned_image': image_aligned}
    
    return alignment


def align_w2(reference, image):
    """
    compute rotational alignment in 2-Wasserstein distance
    input: reference, image = two LxL arrays, pixel values range [0, +], float64
    output: alignment information and aligned image in starting dtype
    """
    
    ny, nx = reference.shape
    angles = np.linspace(0, 360, ny, endpoint=False)

    M = distances_ot.compute_transport_matrix(reference, metric='sqeuclidean')

    dists = distances_ot.rotational_wasserstein_distances(reference, image, M, angles)
    a_min = angles[np.argmin(dists)]
    image_aligned = utils.rotate(image, -a_min)

    alignment = {'angles': angles, 'distances': dists, 'min_theta': a_min, 'aligned_image': image_aligned}
    
    return alignment


def align_sd(reference, image):
    """
    compute rotational alignment in Sinkhorn distance
    input: reference, image = two LxL arrays, pixel values range [0, +], float64
    output: alignment information and aligned image in starting dtype
    """
    
    ny, nx = reference.shape
    angles = np.linspace(0, 360, ny, endpoint=False)

    M = distances_ot.compute_transport_matrix(reference, metric='sqeuclidean')

    dists = distances_ot.rotational_sinkhorn_distances(reference, image, M, angles)
    a_min = angles[np.argmin(dists)]
    image_aligned = utils.rotate(image, -a_min)

    alignment = {'angles': angles, 'distances': dists, 'min_theta': a_min, 'aligned_image': image_aligned}
    
    return alignment


def align_cwd(reference, image):
    """
    compute rotational alignment in convolutional Wasserstein distance
    input: reference, image = two LxL arrays, pixel values range [0, +], float64
    output: alignment information and aligned image in starting dtype
    """
    
    ny, nx = reference.shape
    angles = np.linspace(0, 360, ny, endpoint=False)
    
    dists = distances_ot.rotational_convolutional_wasserstein_distance(reference, image, angles)
    a_min = angles[np.argmin(dists)]
    image_aligned = utils.rotate(image, -a_min)

    alignment = {'angles': angles, 'distances': dists, 'min_theta': a_min, 'aligned_image': image_aligned}
    
    return alignment


def align_wemd(reference, image):
    """
    compute rotational alignment in wavelet Earth mover's distance
    input: reference, image = two LxL arrays, pixel values range [0, +], float64
    output: alignment information and aligned image in starting dtype
    """
    
    ny, nx = reference.shape
    angles = np.linspace(0, 360, ny, endpoint=False)
    
    dists = distances_ot.wemd_rotational_distances(reference, image, angles)
    a_min = angles[np.argmin(dists)]
    image_aligned = utils.rotate(image, -a_min)

    alignment = {'angles': angles, 'distances': dists, 'min_theta': a_min, 'aligned_image': image_aligned}
    
    return alignment


def align_euclidean(reference, image):
    """
    compute rotational alignment in real space Euclidean distance
    input: reference, image = two LxL arrays, pixel values range [0, +], float64
    output: alignment information and aligned image in starting dtype
    """

    ny, nx = reference.shape
    angles = np.linspace(0, 360, ny, endpoint=False)    

    dists = distances.real_space_rotational_distances(reference, image, angles)
    a_min = angles[np.argmin(dists)]
    image_aligned = utils.rotate(image, -a_min)

    alignment = {'angles': angles, 'distances': dists, 'min_theta': a_min, 'aligned_image': image_aligned}
    
    return alignment