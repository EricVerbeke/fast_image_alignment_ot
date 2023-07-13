###

# Authors: 

#==============#
# Notes / Todo #
#==============#


###


import mrcfile
import numpy as np
import finufft
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# from aspire.nufft import nufft
# from aspire.numeric import fft, xp


def log_abs(array):
    return np.log(1 + np.abs(array))


def ft2(array):
    return np.fft.fftshift(np.fft.fft2(array))


def ift2(array):
    return np.fft.ifft2(np.fft.ifftshift(array)).real


def ftn(array):
    return np.fft.fftshift(np.fft.fftn(array))


def iftn(array):
    return np.fft.ifftn(np.fft.ifftshift(array)).real


def rad2degree(radian):
    return radian * 180/np.pi


def rotate(array, degrees, reshape=False):
    return ndi.rotate(array, degrees, reshape=reshape)


def translate(array, shift):
    """Translate a 2D image by pixels, shift is a list for [y, x] translation"""
    return np.roll(array, shift, axis=(0,1))


def get_random_rotations(n_images):
    return np.random.uniform(low=0, high=2*np.pi, size=n_images)


def get_random_translations(n_images, max_x_shift=50, max_y_shift=50):
    # set max shift in integer pixles for now, need better way to do this
    return np.array([(np.random.randint(low=-max_y_shift, high=max_y_shift + 1),
                      np.random.randint(low=-max_x_shift, high=max_x_shift + 1)) for _ in range(n_images)])

def apply_rotations(image, rotations):
    return np.array([rotate(image, r) for r in rotations])


def apply_translations(image, translations):
    return np.array([translate(image, t) for t in translations])


def apply_rotations_then_tranaslations(image, rotations, translations):
    # make this also work for just a single image
    
    assert len(rotations) == len(translations), "number rotations and translations not equal"
    
    images_rt = []
    
    N = len(rotations)
    
    for i in range(N):
        image_r = rotate(image, rotations[i])
        image_rt = translate(image_r, translations[i])
        images_rt.append(image_rt)
        
    return np.array(images_rt)


def open_mrc(mrc_file, return_voxel=False):
    with mrcfile.open(mrc_file) as mrc:
        x = mrc.data
        voxel = mrc.voxel_size.x
        mrc.close()
        if return_voxel:
            x = [x, voxel]
    
    return x


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


def sphere_mask(r_dists, radius=False):
    """Returns sphere mask as boolean"""
    
    if not radius:
        center = [n//2 for n in r_dists.shape]
        radius = np.amin(center)
        
    mask = r_dists <= radius
    
    return mask


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
    V = ftn(v)
    Vb = G * V
    vb = iftn(Vb)
    
    return vb


def fourier_downsample(array, factor=1, rescale=False):
    """Downsample array by cropping its Fourier transform (factor 2 would give 100pix -> 50pix)"""
    
    assert factor >= 1, "scale factor must be greater than 1"
    
    shape = array.shape
    center = [d//2 for d in shape]
    new_shape = [int(d / factor) for d in shape]
    
    F = ftn(array)
    idx = tuple([slice(center[i] - new_shape[i]//2, center[i] + new_shape[i]//2) for i in range(len(shape))])
    F = F[idx] 
    
    if rescale:
        F = F * (np.product(new_shape) / np.product(shape))
    
    f_downsample = iftn(F)
    
    return f_downsample


def fourier_upsample(array, factor=1, rescale=False):
    """Upsample array by zero-padding its Fourier transform (factor 2 would give 100pix -> 200pix)"""
    
    assert factor >= 1, "scale factor must be greater than 1"
    
    shape = array.shape
    
    F = ftn(array)
    p = int((shape[0] * (factor-1)) / 2)
    F_upsample = np.pad(F, p)
    
    if rescale:
        F_upsample = F_upsample * (np.product(F_upsample.shape) / np.product(F.shape))
    
    f_upsample = iftn(F_upsample)
    
    return f_upsample


# def convert_radon_transform_to_cdf(array):
    
#     ny, nx = array.shape
    
#     # rt_matrix = array.copy()  # need this here for now?
        
#     projection_cdfs = [get_cdf_from_pdf(array[:, i]) for i in range(nx)]
    
#     return np.array(projection_cdfs).T


def get_cdf_from_pdf(pdf):
    # assuming the input is strictly positive?
    # assuming projection is pdf and not random var?
    # is this technically cdf or ecdf?
    # fast_radon_transform contains some negatives so need to replace with zero

    # assert np.amin(pdf) >= 0, 'array not positive'
    
    pdf[pdf < 0] = 0   # drop zeros in pdf for now, but need better approach, add as assumption? should handle before here
    # this updates the original array too I think
    
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]

    return cdf


def get_uniform_inverse_cdf_from_pdf(pdf, grid, values):
    # how many grid points should I have, should grid start at 0 or 1/n ?
    
    cdf = get_cdf_from_pdf(pdf)

    inv_cdf = np.interp(grid, cdf, values)
    
    return inv_cdf


def convert_radon_transform_to_inverse_cdf(array, grid, values):
    
    ny, nx = array.shape
        
    projection_inv_cdfs = [get_uniform_inverse_cdf_from_pdf(array[:, i], grid, values) for i in range(nx)]
    
    return np.array(projection_inv_cdfs).T


def normalize_image_to_unit_mass(image):
    # Enforce positive values only and normalize to unit mass
    # - can compare distance between original image after alignment
    
    image[image < 0] = 0
    
    return image / np.sum(image)


# def unit_rescale(vec):
#     """rescale vector values between 0 and 1"""
#     amin = np.amin(vec)
#     amax = np.amax(vec)
    
#     return (vec - amin) / (amax - amin)


# def unit_rescale_matrix_columns(A):
#     """rescale matrix columns between 0 and 1"""
    
#     shape = A.shape
    
#     A_rescale = np.zeros(shape)
    
#     for j in range(shape[1]):
#         A_rescale[:, j] = rescale(A[:, j])
        
#     return A_rescale


def get_sigma_for_snr(x, snr):
    """return standard deviation of WGN for desired snr given real array x"""
    
    N = x.size
    signal = np.sum(x**2)
    noise = np.sqrt(signal / (snr * N))
    
    return noise


def radon_transform_from_real_rotation(array, angles):
    
    # should include a mask after every rotation ?
    
    projections = np.zeros((array.shape[1], len(angles)))
    
    for i, a in enumerate(angles):
        img_r = rotate(array, a)
        p = np.sum(img_r, axis=0)
        projections[:, i] = p
        
    return projections


# def fast_radon_transform(array, angles, use_ramp=False):

#     angles = np.array(angles).flatten()
#     img_size = array.shape[1]
#     rads = angles / 180 * np.pi
#     # y_idx = np.polynomial.chebyshev.chebpts1(img_size)  # use Cheb pts , is this right ?
#     y_idx = np.arange(-img_size / 2, img_size / 2) / img_size * 2
#     x_theta = y_idx[:, np.newaxis] * np.sin(rads)[np.newaxis, :]
#     y_theta = y_idx[:, np.newaxis] * np.cos(rads)[np.newaxis, :]

#     pts = np.pi * np.vstack(
#         [
#             x_theta.flatten(),
#             y_theta.flatten(),
#         ]
#     )
#     pts = pts.astype(np.float32)

#     lines_f = nufft(array, pts).reshape((img_size, -1))

#     if img_size % 2 == 0:
#         lines_f[0, :] = 0

#     # if use_ramp:
#     #     freqs = np.abs(np.pi * y_idx)
#     #     lines_f *= freqs[:, np.newaxis]

#     # projections = np.real(xp.asnumpy(fft.centered_ifft(xp.asarray(lines_f), axis=0)))  # EV: this is slow on my machine
#     projections = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(lines_f, axes=0), axis=0), axes=0).real

#     return projections


def fast_radon_transform(images, angles, nufft_type=2, n_trans=1, eps=1e-8):
    """compute fast radon transform for images using nufft"""
    
    # # for now assume that input is a pair of stacked images to be aligned
    # # need to have check for odd/even shape images
    # # should we use chebyshev points
    
    nz, ny, nx = images.shape
    
    images = images.astype(np.complex128)
    
    rads = angles / 180 * np.pi
    n_lines = len(rads)

    y_idx = np.arange(-ny / 2, ny / 2) / ny * 2
    # y_idx = np.polynomial.chebyshev.chebpts1(ny)

    x_theta = y_idx[:, np.newaxis] * np.sin(rads)[np.newaxis, :]
    x_theta = np.pi * x_theta.flatten()
    
    y_theta = y_idx[:, np.newaxis] * np.cos(rads)[np.newaxis, :]
    y_theta = np.pi * y_theta.flatten()
    
    plan = finufft.Plan(nufft_type, (nx, ny), n_trans, eps)
    plan.setpts(x_theta, y_theta)
    
    img_rts = []
    
    for img in images:
        s = plan.execute(img)
        s = s.reshape(ny, n_lines)
        p = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(s, axes=0), axis=0), axes=0).real
        img_rts.append(p)
        
    return img_rts


def circulant_matvec(x, c):
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(c)).real


def fast_shift_computation(y1, y2):
    # Implement a O(n log n ) method to compute the norm squared difference of the shifts of y1 and y2.
    
    # here each row is a circulant matrix and first
    
    c = np.zeros(y2.size)
    c[0] = y2[0]
    c[1:] = np.flip(y2[1:])

    return np.linalg.norm(y1)**2 + np.linalg.norm(y2)**2 - 2 * circulant_matvec(y1, c)


######################
# Plotting Utilities #
######################

def two_plot(image_1, image_2, size=5, color='viridis'):
    fig, axs = plt.subplots(1, 2, figsize=(size, size))
    axs[0].imshow(image_1, cmap=color)
    axs[0].axis('off')
    axs[1].imshow(image_2, cmap=color)
    axs[1].axis('off')
    fig.tight_layout()
    plt.show()
    

# def compare_line_plots(y1, y2, drop_negative=False):
#     # fix colorbar
    
#     if drop_negative:
#         y1[y1<0] = 0
#         y2[y2<0] = 0
    
#     ny, nx = y1.shape
    
#     grid = np.linspace(0, 1, nx) 
#     values = np.arange(nx)
    
#     p1 = np.sum(y1, axis=0)
#     cdf1 = get_cdf_from_pdf(p1)
#     icdf1 = get_uniform_inverse_cdf_from_pdf(p1, grid, values)
    
#     p2 = np.sum(y2, axis=0)
#     cdf2 = get_cdf_from_pdf(p2)
#     icdf2 = get_uniform_inverse_cdf_from_pdf(p2, grid, values)
    
#     fig, ax = plt.subplots(1, 5, figsize=(14,3))

#     im1 = ax[0].imshow(y1, aspect='equal')
#     ax[0].set_title('image 1')
#     # ax[0].axis('off')
#     ax[0].get_yaxis().set_ticks([])
#     ax[0].get_xaxis().set_ticks([])
#     # plt.colorbar(im1, ax=ax[0], location='left')  
    
#     for side, spine in ax[0].spines.items():
#         spine.set_color('blue')
#         spine.set_linewidth(3)
    
#     # ax[1].set_position(ax[1].get_position())
#     im2 = ax[1].imshow(y2, aspect='equal')
#     ax[1].set_title('image 2')
#     # ax[1].axis('off')
#     ax[1].get_yaxis().set_ticks([])
#     ax[1].get_xaxis().set_ticks([])
#     # plt.colorbar(im2, ax=ax[1], location='left', fraction=0.046)
    
#     for side, spine in ax[1].spines.items():
#         spine.set_color('orange')
#         spine.set_linewidth(3)
    
#     ax[2].plot(p1)
#     ax[2].plot(p2)
#     ax[2].set_title('line projection')
    
#     ax[3].plot(cdf1)
#     ax[3].plot(cdf2)
#     ax[3].set_title('cdf')
    
#     ax[4].plot(grid, icdf1)
#     ax[4].plot(grid, icdf2)
#     ax[4].set_title('inverse cdf')

#     plt.tight_layout(w_pad=0.1)
#     plt.show()



### ========================================================

# def fast_radon_transform(array, angles, use_ramp=False):

#     angles = np.array(angles).flatten()
#     img_size = array.shape[1]
#     rads = angles / 180 * np.pi
#     y_idx = np.polynomial.chebyshev.chebpts1(img_size)  # use Cheb pts ?
#     # y_idx = np.arange(-img_size / 2, img_size / 2) / img_size * 2
#     x_theta = y_idx[:, np.newaxis] * np.sin(rads)[np.newaxis, :]
#     y_theta = y_idx[:, np.newaxis] * np.cos(rads)[np.newaxis, :]

#     pts = np.pi * np.vstack(
#         [
#             x_theta.flatten(),
#             y_theta.flatten(),
#         ]
#     )
#     pts = pts.astype(np.float32)

#     lines_f = nufft(array, pts).reshape((img_size, -1))

#     if img_size % 2 == 0:
#         lines_f[0, :] = 0

#     # if use_ramp:
#     #     freqs = np.abs(np.pi * y_idx)
#     #     lines_f *= freqs[:, np.newaxis]

#     # projections = np.real(xp.asnumpy(fft.centered_ifft(xp.asarray(lines_f), axis=0)))  # EV: this is slow on my machine
#     projections = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(lines_f, axes=0), axis=0), axes=0).real

#     return projections