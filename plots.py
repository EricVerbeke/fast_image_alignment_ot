import numpy as np
import matplotlib.pyplot as plt


def within_n_degrees(gt_rotations, est_rotations, n_deg):
    """number images within n_degrees of ground truth rotation"""
    angle_diffs = abs(gt_rotations - est_rotations)
    within_n = np.count_nonzero(angle_diffs <= n_deg) # want incluse so use leq and geq
    within_n += np.count_nonzero(angle_diffs >= 360-n_deg)
    percent_within_n_degrees = within_n / angle_diffs.size
    
    return percent_within_n_degrees


def two_imshow(image1, image2, cmap='viridis', size=(7,7), cbar=False):
    
    fig, axs = plt.subplots(1, 2, figsize=size)

    img1 = axs[0].imshow(image1, cmap)
    axs[0].axis('off')

    img2 = axs[1].imshow(image2, cmap)
    axs[1].axis('off')
    
    if cbar:
        cbar1 = fig.colorbar(img1, ax=axs[0], aspect=size[0])
        cbar2 = fig.colorbar(img2, ax=axs[1], aspect=size[0])

    plt.tight_layout()
    plt.show()
    

def square_tile_plot(images, length, spine_colors, spine_size=1, size=5, pad=0.3, cmap='gray', save_path=False, show=True):
    
    # assert images.shape[0] >= length**2, "not enough images"
    N, ny, nx = images.shape
    
    fig, axs = plt.subplots(length, length, figsize=(size,size))
    axs = axs.flatten()
    
    for idx, ax in enumerate(axs):
        if idx < N:
            ax.imshow(images[idx], cmap=cmap)
            # ax.axis('off')
            
        else:
            ax.imshow(np.zeros((ny, nx)), cmap=cmap)
        
        for side, spine in ax.spines.items():
            spine.set_color(spine_colors[idx])
            spine.set_linewidth(spine_size)
            
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        
    plt.tight_layout(h_pad=pad, w_pad=pad)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

        
def rectangular_tile_plot(images, rows, cols, spine_colors, spine_size=1, size_x=5, size_y=3,
                          hpad=0.3, wpad=0.3, cmap='gray', save_path=False, show=True):
        
    fig, axs = plt.subplots(rows, cols, figsize=(size_x, size_y))
    axs = axs.flatten()
    
    for idx, ax in enumerate(axs):
        ax.imshow(images[idx], cmap=cmap)
        # ax.axis('off')
        
        for side, spine in ax.spines.items():
            spine.set_color(spine_colors[idx])
            spine.set_linewidth(spine_size)
            
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        
    plt.tight_layout(h_pad=hpad, w_pad=wpad)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

        
def line_tile_plot(images, length, spine_colors, spine_size=1, size=5, pad=0.3, cmap='gray', save_path=False, show=True):
        
    fig, axs = plt.subplots(1, length, figsize=(size,size))
    axs = axs.flatten()
    
    for idx, ax in enumerate(axs):
        ax.imshow(images[idx], cmap=cmap)
        
        for side, spine in ax.spines.items():
            spine.set_color(spine_colors[idx])
            spine.set_linewidth(spine_size)
            
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])
        
    plt.tight_layout(w_pad=pad)
    
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()