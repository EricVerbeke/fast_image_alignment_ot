import gzip
import numpy as np

from scipy import ndimage as ndi


def log_abs(array):
    return np.log(1 + np.abs(array))


def generate_centered_gaussian(L=128, d=2, sigma=1):
    
    l = np.linspace(-L/2, L/2, L)
    p = np.array(np.meshgrid(*[l for _ in range(d)]))
    g = np.exp(-( np.sum(p**2, axis=0) ) / (2*sigma**2))
    g = g / np.sum(g)  # normalize to sum = 1
    
    return g


def signal_convolution(f, g):
    """convolve image f with g, both real LxL arrays"""

    ### note: doesn't account for circular convolution
    
    g_hat = np.fft.fftshift(np.fft.fftn(g))
    f_hat = np.fft.fftshift(np.fft.fftn(f))
    z = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(g_hat * f_hat))).real

    return z


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


def translate(array, ty, tx):

    return np.roll(array, (ty, tx), axis=(0,1))


def zero_pad_image_stack(image_stack, w):
    
    return np.pad(image_stack, pad_width=((0, 0), (w, w), (w, w)))


def zero_pad_image_stack_to_size(image_stack, size):

    if len(image_stack.shape) == 2:
        image_stack = image_stack[np.newaxis, ...]
        
    N, ny, nx = image_stack.shape

    d = size - ny

    assert d > 0, 'new size is smaller than original'

    if d % 2 == 0:
        w = (size - ny) // 2
        image_stack = np.pad(image_stack, pad_width=((0, 0), (w, w), (w, w)))

    else:
        w = (size - ny) / 2
        if w < 1:
            image_stack = np.pad(image_stack, pad_width=((0, 0), (0, 1), (0, 1)))
        else:
            l = int(w - 0.5)
            r = int(w + 0.5)
            image_stack = np.pad(image_stack, pad_width=((0, 0), (l, r), (l, r))) 
            
    return image_stack


def random_points_in_sphere_voxelized(L, D, n):
    """Generate n random points uniformly distributed inside a sphere of given radius"""
    
    V = np.zeros((L, L, L))
    radius = D / L
    
    phi = np.random.uniform(0, 2 * np.pi, n)   # azimuthal angle
    costheta = np.random.uniform(-1, 1, n)     # cosine of polar angle
    u = np.random.uniform(0, 1, n)             # random variable for radius

    theta = np.arccos(costheta)                # convert to polar angle
    r = radius * (u ** (1/3))                  # cube root to maintain uniformity

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    points = np.column_stack((x, y, z))
    voxels = (np.round((points * L//2)) + L//2).astype('int')  # nearest neighbor interpolation
    V[tuple(voxels.T)] = 1
    
    return V


def generate_projections(V, angles, axis=0):
    """"Function to generate 2D projection images"""
    
    projections = []
    for angle in angles:
        rotated_V = ndi.rotate(V, angle, axes=(0, 2), reshape=False, mode='nearest')
        projection = np.sum(rotated_V, axis=axis)  # Sum along the chosen axis
        projections.append(projection)
    return np.array(projections)


def load_mnist_images(image_file):
    with gzip.open(image_file, 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big') # first 4 bytes is a magic number
        image_count = int.from_bytes(f.read(4), 'big')  # second 4 bytes is the number of images
        row_count = int.from_bytes(f.read(4), 'big') # third 4 bytes is the row count
        column_count = int.from_bytes(f.read(4), 'big') # fourth 4 bytes is the column count
        image_data = f.read() # rest is the image pixel data stored as an unsigned byte, values are 0 to 255        
        images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
        
        return images
    
    
def load_mnist_labels(label_file):
    with gzip.open(label_file, 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big') # first 4 bytes is a magic number
        label_count = int.from_bytes(f.read(4), 'big') # second 4 bytes is the number of labels
        label_data = f.read() # rest is the label data stored as unsigned byte, values are 0 to 9
        labels = np.frombuffer(label_data, dtype=np.uint8)
        
        return labels


# def fourier_pad_image_stack(imgs, p=2):
#     shape = imgs.shape
#     l = shape[1] * p
#     imgs_ft = np.fft.fftshift(np.fft.fft2(imgs))
#     imgs_ft_pad = zero_pad_image_stack_to_size(imgs_ft, l)
#     imgs_pad = np.fft.ifft2(np.fft.ifftshift(imgs_ft_pad)).real

#     return imgs_pad


# def distance_matrix_from_dict(dists_dict, n_images):
    
#     dist_mat = np.zeros((n_images, n_images))
    
#     for key, dist in dists_dict.items():
        
#         idx1, idx2 = key
#         dist_min = np.amin(dist)
#         dist_mat[idx1, idx2] = dist_min
#         dist_mat[idx2, idx1] = dist_min
        
#     return dist_mat