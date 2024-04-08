import numpy as np
import matplotlib.pyplot as plt


def within_n_degrees(gt_rotations, est_rotations, n_deg):
    """number images within n_degrees of ground truth rotation"""
    angle_diffs = abs(gt_rotations - est_rotations)
    within_n = np.count_nonzero(angle_diffs <= n_deg) # want incluse so use leq and geq
    within_n += np.count_nonzero(angle_diffs >= 360-n_deg)
    percent_within_n_degrees = within_n / angle_diffs.size
    
    return percent_within_n_degrees


def two_imshow(image1, image2, size=(7,7), cbar=False):
    
    fig, axs = plt.subplots(1, 2, figsize=size)

    img1 = axs[0].imshow(image1)
    axs[0].axis('off')

    img2 = axs[1].imshow(image2)
    axs[1].axis('off')
    
    if cbar:
        cbar1 = fig.colorbar(img1, ax=axs[0], aspect=size[0])
        cbar2 = fig.colorbar(img2, ax=axs[1], aspect=size[0])

    plt.tight_layout()
    plt.show()
    
    
def square_tile_plot(images, length, size=5, pad=0.3, cmap='gray', save_path=False, show=True):
    
    assert images.shape[0] >= length**2, "not enough images"
    
    fig, axs = plt.subplots(length, length, figsize=(size,size))
    axs = axs.flatten()
    
    for idx, ax in enumerate(axs):
        ax.imshow(images[idx], cmap=cmap)
        ax.axis('off')
        
    plt.tight_layout(h_pad=pad, w_pad=pad)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
        
        
def line_tile_plot(images, length, size=5, pad=0.3, cmap='gray', save_path=False, show=True):
        
    fig, axs = plt.subplots(1, length, figsize=(size,size))
    axs = axs.flatten()
    
    for idx, ax in enumerate(axs):
        ax.imshow(images[idx], cmap=cmap)
        ax.axis('off')
        
    plt.tight_layout(h_pad=pad, w_pad=pad)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
