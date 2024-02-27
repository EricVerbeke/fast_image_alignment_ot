import numpy as np


def circulant_matvec(v, c):

    return np.fft.ifft(np.fft.fft(v) * np.fft.fft(c)).real


def fast_shift_computation(vi, vj):
    ### compute the norm squared difference of the shifts of vi and vj in O(n log n).
    ### i.e., the circular convolution for each row vector

    c = np.zeros(vj.size).astype(vj.dtype)  # inherit complex for complex L2
    c[0] = vj[0]
    c[1:] = np.flip(vj[1:])
    dist = np.linalg.norm(vi)**2 + np.linalg.norm(vj)**2 - 2*circulant_matvec(vi, np.conj(c))

    return dist


def rotational_distances(image1, image2, n_points):
    
    dist_matrix = np.array([fast_shift_computation(image1[j, :], image2[j, :]) for j in range(n_points)])
    dists = np.sum(dist_matrix, axis=0)
    
    return dists


def signed_rotational_distances(image1_pos, image2_pos, image1_neg, image2_neg, n_points):
    
    dists_pos = rotational_distances(image1_pos, image2_pos, n_points)
    dists_neg = rotational_distances(image1_neg, image2_neg, n_points)
    
    dists = dists_pos + dists_neg
    
    return dists


### not exactly sure how to set up these funcitons
### should signed be separate function? should include all scores or just min?
### change formatting from dict to matrix? add function to compute mins
def reference_rotational_distances(reference, images, n_points):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        dists_dict[i] = rotational_distances(reference, images[i], n_points)
        
    return dists_dict
    
    
def signed_reference_rotational_distances(ref_pos, images_pos, ref_neg, images_neg, n_points):
    
    N = images_pos.shape[0]
    dists_dict = {}
    
    for i in range(N):
        dists_dict[i] = signed_rotational_distances(ref_pos, images_pos[i], ref_neg, images_neg[i], n_points)
        
    return dists_dict
        
        
def pairwise_rotational_distances(images, n_points):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = rotational_distances(images[i], images[j], n_points)
            
    return dists_dict


def signed_pairwise_rotational_distances(images_pos, images_neg, n_points):
    
    N = images_pos.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = signed_rotational_distances(images_pos[i], images_pos[j], images_neg[i], images_neg[j], n_points)
            
    return dists_dict


def translate(array, ty, tx):

    return np.roll(array, (ty, tx), axis=(0,1))


def fast_translation_distance(image1, image2):
    ### note assumptions about this method e.g. circular convolution
    ### vectorize this to multiple images
    
    image1_ft = np.fft.fft2(image1)
    image2_ft = np.fft.fft2(image2)

    corr = np.fft.ifft2(np.conj(image1_ft) * image2_ft)
    ty, tx = np.unravel_index(np.argmax(corr), corr.shape)
    
    return ty, tx


def shift_align_images(image1, image2):
    ### change this function name
    
    ty, tx = fast_translation_distance(image1, image2)
    
    image_2_align = translate(image2, -ty, -tx)
    
    return image_2_align