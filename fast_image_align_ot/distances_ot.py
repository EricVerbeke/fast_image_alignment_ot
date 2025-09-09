### standard libraries
import numpy as np

### OT libraries
import pywt 
import ot 

import jax
import jax.numpy as jnp

from ott.geometry import costs, grid, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

# ### other libraries
# import ftk  

### my libraries
from fast_image_align_ot import utils


#################################################
#####            OTHER DISTANCES            #####
#####          W2 / SD / CW / WEMD          #####
#################################################

### Compute various transport distances between images

### (W2): 2-Wasserstein
### (SD): Sinkhorn distance
### (CW): Convolutional Wasserstein distance
### (WEMD): wavelet Earth mover's distance


def compute_transport_matrix(image, metric='sqeuclidean'):
    """
    when using emd2 for Wasserstein distance, 
    metric='sqeuclidean' -> W2 and metric='euclidean' -> W1
    https://pythonot.github.io/quickstart.html
    """
    
    ny, nx = image.shape
    xs = np.arange(nx)
    xgrid, ygrid = np.meshgrid(xs, xs)
    points = np.array(list(zip(ygrid.flatten(), xgrid.flatten())))  # order matches flatten convention
    M = ot.dist(points, points, metric=metric)  # returns the 2-norm squared, change metric='euclidean' for w1    
    M = M / np.amax(M)  
    
    return M


def wasserstein_distance(image1, image2, M, numItermax=100000):
        
    image1_flat = image1.flatten()
    image1_flat = np.where(image1_flat<0, 0, image1_flat)
    image1_flat *= 1.0 / image1_flat.sum()
    
    image2_flat = image2.flatten()
    image2_flat = np.where(image2_flat<0, 0, image2_flat)
    image2_flat *= 1.0 / image2_flat.sum()
    
    dist = ot.emd2(image1_flat, image2_flat, M, numItermax=numItermax)

    return dist


def rotational_wasserstein_distances(image1, image2, M, angles, numItermax=100000):
        
    dists = []
    
    image1 = np.where(image1<=0, 0, image1)
    image1 *= 1.0 / image1.sum()
    image1_flat = image1.flatten()

    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        image2_rot = np.where(image2_rot<0, 0, image2_rot)
        image2_rot *= 1.0 / image2_rot.sum()  # normalize images back to one
        image2_flat = image2_rot.flatten()

        dists.append(ot.emd2(image1_flat, image2_flat, M, numItermax=numItermax))  # EV: add numItermax?

    return np.array(dists)


def sinkhorn_distance(image1, image2, M, reg=0.01, numItermax=3):

    epsilon_mass = 1e-10

    image1 = np.where(image1<=0, epsilon_mass, image1)
    image1 *= 1.0 / image1.sum()
    image1_flat = image1.flatten()

    image2 = np.where(image2<=0, epsilon_mass, image2)
    image2 *= 1.0 / image2.sum()
    image2_flat = image2.flatten()

    dist = ot.sinkhorn2(image1_flat, image2_flat, M, reg=reg, numItermax=numItermax)

    return dist


def rotational_sinkhorn_distances(image1, image2, M, angles, reg=0.01, numItermax=3):

    epsilon_mass = 1e-10 # add small mass to avoid zeros
        
    dists = []
     
    image1 = np.where(image1<=0, epsilon_mass, image1)
    image1 *= 1.0 / image1.sum()
    image1_flat = image1.flatten()

    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        image2_rot = np.where(image2_rot<=0, epsilon_mass, image2_rot)  # catch negative values from interpolation
        image2_rot *= 1.0 / image2_rot.sum()  # normalize images back to one
        image2_flat = image2_rot.flatten()

        dists.append(ot.sinkhorn2(image1_flat, image2_flat, M, reg=reg, numItermax=numItermax))

    return np.array(dists)


def convolutional_wasserstein_distance(image1, image2, epsilon=0.01, max_iterations=3):

    epsilon_mass = 1e-10

    grid_size = image1.shape
    geom = grid.Grid(grid_size=grid_size, epsilon=epsilon)

    image1_flat = image1.flatten()
    image1_flat = np.where(image1_flat<=0, epsilon_mass, image1_flat)
    image1_flat *= 1.0 / image1_flat.sum()
    image1_flat += epsilon_mass ### EV: this seems to prevent error
    
    image2_flat = image2.flatten()
    image2_flat = np.where(image2_flat<=0, epsilon_mass, image2_flat)
    image2_flat *= 1.0 / image2_flat.sum()
    image2_flat += epsilon_mass ### EV: this seems to prevent error

    prob = linear_problem.LinearProblem(geom, a=image1_flat, b=image2_flat)
    solver = sinkhorn.Sinkhorn(max_iterations=max_iterations)
    out = solver(prob)

    dist = out.reg_ot_cost
    
    return dist


def rotational_convolutional_wasserstein_distance(image1, image2, angles, epsilon=0.01, max_iterations=3):

    dists = []

    epsilon_mass = 1e-10

    grid_size = image1.shape
    geom = grid.Grid(grid_size=grid_size, epsilon=epsilon)
    solver = sinkhorn.Sinkhorn(max_iterations=max_iterations)

    image1_flat = image1.flatten()
    image1_flat = np.where(image1_flat<=0, epsilon_mass, image1_flat)
    image1_flat *= 1.0 / image1_flat.sum()
    image1_flat += epsilon_mass ### EV: this seems to prevent error    

    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        image2_rot = np.where(image2_rot<=0, epsilon_mass, image2_rot)
        image2_rot *= 1.0 / image2_rot.sum()  # normalize images back to one
        image2_flat = image2_rot.flatten()
        image2_flat += epsilon_mass

        prob = linear_problem.LinearProblem(geom, a=image1_flat, b=image2_flat)
        out = solver(prob)

        dists.append(out.reg_ot_cost)

    return np.array(dists)


def embed(arr, wavelet, level):
    """
    *** This code is borrowed from: https://github.com/RuiyiYang/BOTalign/blob/main/wemd.py#L11
    This function computes an embedding of Numpy arrays such that the L1 distance
    between the resulting embeddings is approximately equal to the Earthmover distance of the arrays.

    Input:
        arr - numpy array.
        level - Decomposition level of the wavelets. Larger levels yield more coefficients and more accurate results.
        wavelet - either the name of a wavelet supported by PyWavelets (e.g. 'coif3', 'sym3') or a pywt.Wavelet object.
                  See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html#built-in-wavelets-wavelist
    Output:
        One-dimensional numpy array containing weighted details coefficients.
        Approximate Earthmover's distances are then given by the l_1 distances of the results, e.g.
            wemd(arr1, arr2) := numpy.linalg.norm(embed(arr1, wavelet, level)-embed(arr2, wavelet, level), ord=1)
    """

    arrdwt = pywt.wavedecn(arr/arr.sum(), wavelet, mode='zero', level=level)

    dimension = len(arr.shape)
    assert dimension in [2,3]

    n_levels = len(arrdwt[1:])

    weighted_coefs = [arrdwt[0].flatten()*2**(n_levels)]
    for (j, details_level_j) in enumerate(arrdwt[1:]):
        for coefs in details_level_j.values():
            multiplier = 2**((n_levels-1-j)*(1+(dimension/2.0)))
            weighted_coefs.append(coefs.flatten()*multiplier)

    return np.concatenate(weighted_coefs)


def wemd_distance(image1, image2, wavelet='sym3', level=3):

    w1 = embed(image1, wavelet, level)
    w2 = embed(image2, wavelet, level)

    return np.linalg.norm(w1 - w2, ord=1)


def wemd_rotational_distances(image1, image2, angles, wavelet='sym3', level=3):
        
    dists = []
    
    w1 = embed(image1, wavelet, level)
    
    for a in angles:
        image2_rot = utils.rotate(image2, -a)
        w2 = embed(image2_rot, wavelet, level)
        dists.append(np.linalg.norm(w1 - w2, ord=1))
        
    return np.array(dists)



# def wasserstein_distance(image1, image2, M, numItermax=100):

#     epsilon_mass = 1e-10
        
#     image1_flat = image1.flatten()
#     image1_flat = np.where(image1_flat<=0, epsilon_mass, image1_flat)
#     image1_flat *= 1.0 / image1_flat.sum()
    
#     image2_flat = image2.flatten()
#     image2_flat = np.where(image2_flat<=0, epsilon_mass, image2_flat)
#     image2_flat *= 1.0 / image2_flat.sum()
    
#     dist = ot.emd2(image1_flat, image2_flat, M, numItermax)

#     return dist


# def rotational_wasserstein_distances(image1, image2, M, angles, numItermax=100000):
        
#     dists = []
    
#     epsilon_mass = 1e-8   # add small mass to avoid zeros
    
#     image1 = image1 + epsilon_mass
#     image1 *= 1.0 / image1.sum()
#     image1_flat = image1.flatten()

#     for a in angles:
#         image2_rot = utils.rotate(image2, -a)
#         image2_rot = np.where(image2_rot<=0, epsilon_mass, image2_rot)
#         image2_rot *= 1.0 / image2_rot.sum()   # normalize images back to one
#         image2_flat = image2_rot.flatten()

#         dists.append(ot.emd2(image1_flat, image2_flat, M, numItermax))  # EV: add numItermax?

#     return np.array(dists)




### FTK ###


# def ftk_precompute(reference, images, n_psi, B):
    
#     L = reference.shape[0]
#     n_images = images.shape[0]
    
#     T = 2
#     N = L
#     ftk_eps = 1e-2
    
#     delta_range = 1*B
#     oversampling = 1  # what should this parameter be???
    
#     rmax = N / 2*np.pi
#     ngridr = 3*N
    
#     # the following four lines of code perform the necessary precompution for the fourier grids
#     pf_grid = ftk.make_tensor_grid(rmax, ngridr, n_psi) # grid for polar fourier transform
#     tr_grid = ftk.make_adaptive_grid(delta_range, T/N, oversampling) # grid for shift
#     n_bessel = int(np.ceil(delta_range))
#     plan = ftk.ftk_plan(tr_grid, pf_grid, n_bessel, ftk_eps)
    
#     # the following code computes the polar fourier transforms of the images 
#     Shat = ftk.cartesian_to_pft(images, T, pf_grid)
#     Mhat = ftk.cartesian_to_pft(reference.reshape((1,L,L)), T, pf_grid)
    
#     prods_ftk, tm_ftk = ftk.ftk_execute(plan, Mhat, Shat)
    
#     ftk_parameters = [n_images, prods_ftk, pf_grid, n_psi, tr_grid]
    
#     return ftk_parameters
    

# def ftk_align(reference, images, n_psi, B):
    
#     ftk_parameters = ftk_precompute(reference, images, n_psi, B)
#     n_images = ftk_parameters[0]
#     prods_ftk = ftk_parameters[1]
#     pf_grid = ftk_parameters[2]
#     n_psi = ftk_parameters[3]
#     tr_grid = ftk_parameters[4]
    
#     corr=np.zeros(n_images)  # the maximum correlation found (theoretical maximum should be 1)
#     angle_rec=np.zeros(n_images)  # the in-plane rotation angle recovered
#     shift_rec=np.zeros((n_images,2))  # the shift recovered
    
#     for idx in range(n_images):      
#         index=np.unravel_index(prods_ftk[0,idx,:,:].argmax(), prods_ftk[0,idx,:,:].shape)
#         corr[idx]=prods_ftk[0,idx,:,:].max()
#         angle_rec[idx]=pf_grid['all_psi'][:n_psi][index[0]]*180/np.pi 
#         shift_rec[idx]=tr_grid['trans'][index[1]]
    
#     return corr, angle_rec, shift_rec