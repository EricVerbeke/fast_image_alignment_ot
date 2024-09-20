import gzip
import numpy as np
from scipy import ndimage as ndi


def log_abs(array):
    return np.log(1 + np.abs(array))


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


