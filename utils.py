import numpy as np
from scipy import ndimage as ndi


def radial_distance_grid(shape):
    """Compute grid of radial distances"""
    
    center = [n//2 for n in shape]
    idx = [slice(-center[i], l-center[i]) for i, l in enumerate(shape)] 
    coords = np.ogrid[idx] # zero-centered grid index
    square_coords = [c**2 for c in coords] # square grid for distance (x^2 + y^2 + z^2 = r^2)
    
    radial_dists = square_coords[0] # initialize to broadcast distance grid by dimension
    for dimension in range(1, len(shape)):
        radial_dists = radial_dists + square_coords[dimension]
        
    return np.round(np.sqrt(radial_dists))


def sphere_mask(rdists, radius=False):
    """Returns sphere mask as boolean"""
    
    if not radius:
        center = [n//2 for n in rdists.shape]
        radius = np.amin(center)
        
    mask = rdists <= radius
    
    return mask


def rotate(array, degrees, reshape=False):
    
    return ndi.rotate(array, degrees, reshape=reshape)


##### add/test these functions later

# def real_space_rotational_distance(img_1, img_2, angles, all_scores=False):
    
#     dists = []
    
#     for a in angles:
#         img_2_rot = ndi.rotate(img_2, -a, order=3, reshape=False)  # rotate clockwise as hack for now
#         dists.append(np.linalg.norm(img_1 - img_2_rot)**2)  # squared-norm to match fast conv
        
#     d_min_idx = np.argmin(dists)
#     d_min = dists[d_min_idx]
#     a_min = angles[d_min_idx]
    
#     if all_scores:
#         return dists
#     else:
#         return d_min, a_min
    

# def radon_transform_from_skimage(array, angles):
#     from skimage.transform import radon
#     return radon(array, angles, preserve_range=True)[:, ::-1]


# def radon_transform_from_real_rotation(array, angles):
        
#     projections = np.zeros((array.shape[1], len(angles)))
    
#     for i, a in enumerate(angles):
#         img_r = rotate(array, a)
#         p = np.sum(img_r, axis=0)
#         projections[:, i] = p
        
#     return projections