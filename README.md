# Fast alignment of heterogeneous images in sliced Wasserstein distance

<img align="left" width="800" src="https://github.com/EricVerbeke/fast_image_alignment_ot/blob/main/Figures/demo.pdf"/>

<br clear="left"/><br/>

An algorithm for the fast computation of the sliced 2-Wasserstein distance between two images. Can be used for rotational alignment of two $$L \times L$$ images in $$O(L^2 \log L)$$ operations.
This algorithm is shown to be robust to rotations, translations and deformations in the images.

See (upcoming manuscript) for more details about the algorithm. Tutorial on how to use will follow soon.

Dependencies:
- numpy
- matplotlib
- scipy
- finufft
- pywavelets
- pot
