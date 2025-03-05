import numpy as np

### my lib
import utils


class Image:
    
    def __init__(self, images, normalize=True, mask=True, radius=False, scale=False, add_noise=False, snr=1):
        """
        Preprocess and various image modifications
        
        TODO:
        - add Fourier scaling / cropping
        - control image dtype
        """
        self.images = images
        self.shape = images.shape
        self.normalize = normalize
        self.mask = mask
        self.radius = radius
        self.scale = scale
        self.add_noise = add_noise
        self.snr = snr
        
        if len(self.shape) == 2:
            self.N, self.ny, self.nx = 1, self.shape[0], self.shape[1]
            self.images = images[np.newaxis, ...]
        elif len(self.shape) == 3:
            self.N, self.ny, self.nx = self.shape[0], self.shape[1], self.shape[2]
        else:
             raise ValueError("input must be LxL image, or NxLxL image stack")
        assert self.ny == self.nx, "images should be size L x L"
        
        
    def apply_mask(self):
        
        rdists = utils.radial_distance_grid([self.ny, self.nx])
        mask = utils.sphere_mask(rdists, radius=self.radius)
        images_mask = mask * self.images
        
        return images_mask
    
        
    def normalize_images(self):
        """normalize to unit mass"""
        
        scale = np.sum(self.images, axis=(1,2))
        images_norm = self.images / scale.reshape(self.N, 1, 1)  
        
        return images_norm
    

# ### add these options later
#     def fourier_downsample(self):
#         pass

    
#     def fourier_upsample(self):
#         pass
    

#     def fourier_scale_images(self):
#         ### check function for non-integer scaling
#         # assert self.scale  ## is a number, >0, etc
        
#         if self.scale > 1:
#             images_scale = self.fourier_upsample()
#         elif self.scale < 1:
#             images_scale = self.fourier_downsample()
        
#         return images_scale
# ###
   
    
    def _add_white_gaussian_noise(self):
        
        signal = np.sum(self.images**2, axis=(1,2))
        noise_var = signal / (self.snr * self.ny * self.nx)
        noise_std = np.sqrt(noise_var).reshape(self.N, 1, 1)

        additive_noise = np.random.normal(0, noise_std, (self.shape))
        images_noise = self.images + additive_noise
    
        return images_noise
    
    
    def preprocess_images(self):
        
        # if self.scale:
        #     self.images = self.fourier_scale_images()
            
        if self.add_noise:
            self.images = self._add_white_gaussian_noise()
        
        if self.mask:
            self.images = self.apply_mask()
            
        if self.normalize:
            self.images = self.normalize_images()
            
        return self.images