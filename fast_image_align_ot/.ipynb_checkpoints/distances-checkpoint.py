import numpy as np

### my lib
from fast_image_align_ot import utils


def rotational_distances(U, V):
    """fast compute ||U - T_l V||^2_F for all column shifts of V"""
    
    U_norm = np.linalg.norm(U)**2
    V_norm = np.linalg.norm(V)**2

    C = np.zeros(V.shape).astype(V.dtype)
    C[:, 0] = V[:, 0]
    C[:, 1:] = np.flip(V[:, 1:], axis=1)

    Uj_hat = np.fft.fft(U, axis=1)
    Cj_hat = np.fft.fft(C, axis=1)

    UV_l = np.fft.ifft(Uj_hat * Cj_hat, axis=1).real
    UV_l_sum = np.sum(UV_l, axis=0)

    dists = U_norm + V_norm - 2*UV_l_sum
    
    return dists


def signed_rotational_distances(Up, Vp, Un, Vn):
    
    dists_pos = rotational_distances(Up, Vp)
    dists_neg = rotational_distances(Un, Vn)
    
    dists = (dists_pos + dists_neg) / 2
        
    return dists


def l2_distance(U, V):
    
    return np.linalg.norm(U - V)**2


def sliced_distance(U, V):
    
    d_w2 = np.mean((U - V)**2, axis=0)
    d_sw2 = np.mean(d_w2)
    
    return d_sw2


def signed_sliced_distance(Up, Vp, Un, Vn):
    
    d_sw2_p = sliced_distance(Up, Vp)
    d_sw2_n = sliced_distance(Un, Vn)
    
    d_sw2 = (d_sw2_p + d_sw2_n) / 2
    
    return d_sw2


def max_sliced_distance(U, V):
    
    d = np.mean((U - V)**2, axis=0)
    d_max = np.amax(d)
    
    return d_max


def rotational_max_sliced_wasserstein(U, V, n_theta):
        
    return np.array([max_sliced_distance(U, utils.translate(V, 0, l)) for l in range(n_theta)])


def reference_rotational_max_sliced_wasserstein(U, V, n_theta, N):

    return {idx: rotational_max_sliced_wasserstein(U, V[idx], n_theta) for idx in range(N)}


def signed_rotational_max_sliced_wasserstein(Up, Vp, Un, Vn, n_theta):

    d_rf_msw2_p = rotational_max_sliced_wasserstein(Up, Vp, n_theta)
    d_rf_msw2_n = rotational_max_sliced_wasserstein(Un, Vn, n_theta)

    d_rf_msw2 = (d_rf_msw2_p + d_rf_msw2_n) / 2

    return d_rf_msw2


def reference_signed_rotational_max_sliced_wasserstein(Up, Vp, Un, Vn, n_theta, N):

    return {idx: signed_rotational_max_sliced_wasserstein(Up, Vp[idx], Un, Vn[idx], n_theta) for idx in range(N)}
    
            
def reference_rotational_distances(reference, images):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        dists_dict[i] = rotational_distances(reference, images[i])
        
    return dists_dict
    
    
def reference_signed_rotational_distances(ref_pos, images_pos, ref_neg, images_neg):
    
    N = images_pos.shape[0]
    dists_dict = {}
    
    for i in range(N):
        dists_dict[i] = signed_rotational_distances(ref_pos, images_pos[i], ref_neg, images_neg[i])
        
    return dists_dict

   
def pairwise_rotational_distances(images, n_points):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = rotational_distances(images[i], images[j], n_points)
            
    return dists_dict


def pairwise_signed_rotational_distances(images_pos, images_neg):
    
    N = images_pos.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = signed_rotational_distances(images_pos[i], images_pos[j], images_neg[i], images_neg[j])
            
    return dists_dict


def real_space_rotational_distances(image1, image2, angles):
    
    dists = []
    
    for a in angles:
        image2_rot = utils.rotate(image2, -a)  # rotate clockwise as hack for now
        dists.append(np.linalg.norm(image1 - image2_rot)**2)  # squared-norm to match fast conv
        
    return np.array(dists)


def real_space_rotational_distance_pairwise(images, angles):
    
    N = images.shape[0]
    dists_dict = {}
    
    for i in range(N):
        for j in range(i+1, N):
            dists_dict[(i, j)] = real_space_rotational_distance(images[i], images[j], angles)
            
    return dists_dict


def fast_translational_distances(image1, image2):
    ### does not zero pad (i.e. circular translation allowed)
    ### vectorize this to multiple images
    
    image1_ft = np.fft.fft2(image1)
    image2_ft = np.fft.fft2(image2)

    corr = np.fft.ifft2(np.conj(image1_ft) * image2_ft)
    ty, tx = np.unravel_index(np.argmax(corr), corr.shape)
    
    return ty, tx

    
# def slow_sliced_rotational_distances(U, V, n_theta):
    
#     dists = np.zeros(n_theta)

#     for l in range(n_theta):

#         d_w2 = np.mean((U - utils.translate(V, 0, l))**2, axis=0)
#         d_sw2 = np.mean(d_w2)

#         dists[l] = d_sw2
        
#     return dists


# def slow_signed_sliced_rotational_distances(Up, Vp, Un, Vn, n_theta):
            
#     d_sw2_p = slow_sliced_wasserstein_rotation(Up, Vp, n_theta)
#     d_sw2_n = slow_sliced_wasserstein_rotation(Un, Vn, n_theta)

#     dists = (d_sw2_p + d_sw2_n) / 2

#     return dists