import multiprocessing
from IPython.display import clear_output
from typing import NamedTuple

import numpy as np
import pandas as pd

from .morpho import bounding_parallelogram, ImageMoments, ImageMorphology


class Morphometrics(NamedTuple):
    """Measured shape attributes of an image.

    - area: Total area/image mass.
    - length: Length of the estimated skeleton.
    - thickness: Mean thickness along the skeleton.
    - slant: Horizontal shear, in radians.
    - width: Width of the bounding parallelogram.
    - height: Height of the bounding parallelogram.
    """
    area: float
    length: float
    thickness: float
    slant: float
    width: float
    height: float


def measure_image(image, threshold: float = .5, scale: int = 4, bound_frac: float = .02,
                  verbose=True):
    """Computes morphometrics for a single image.

    Parameters
    ----------
    image : (H, W) array_like
        Input image.
    threshold : float, optional
        A relative threshold between 0 and 1. The upsampled image will be binarised at this fraction
        between its minimum and maximum values.
    scale : int, optional
        Upscaling factor for subpixel morphological analysis (>=1).
    bound_frac : float, optional
        Fraction of image mass to discard along each dimension when computing the bounding
        parallelogram.
    verbose : bool, optional
        Whether to pretty-print the estimated morphometrics.

    Returns
    -------
    Morphometrics
        A namedtuple containing the measured area, length, thickness, slant, width, and height.
    """
    image = np.asarray(image)
    morph = ImageMorphology(image, threshold, scale)
    moments = ImageMoments(morph.hires_image)
    thickness = morph.mean_thickness
    area = morph.area
    length = morph.stroke_length
    large = morph.stroke_large
    slant = np.arctan(-moments.horizontal_shear)
    
    corners = bounding_parallelogram(morph.hires_image, bound_frac, moments)
    width = (corners[1][0] - corners[0][0]) / morph.scale
    height = (corners[-1][1] - corners[0][1]) / morph.scale

    if verbose:
        print(f"Area: {area:.1f}")
        print(f"Length: {length:.1f}")
        print(f"Thickness: {thickness:.2f}")
        print(f"Slant: {np.rad2deg(slant):.0f}°")
        print(f"Dimensions: {width:.1f} x {height:.1f}")

    return area, thickness, slant, width, height


def get_Y_disc_from_X(X, threshold = 0.5, scale = 4, batchsize = 10) :

    def measure_batch(X) :
        return pd.Series(np.array(measure_image(X, threshold = threshold, scale=scale, verbose = 0))) # precision  threshold = 0.35, scale=10,
    
    batch_range = range(0, X[:].shape[0] +1, batchsize)
    
    Y_cont_verif = []
    for i, batch in enumerate(batch_range[:-1]) :
        clear_output()
        print(f"Epoch {i+1}/{len(batch_range) -1}")
        Y_cont_verif.append(pd.DataFrame(map(measure_batch, X[batch:batch_range[i +1], 0])))

    Y_cont_verif = pd.concat(Y_cont_verif) 
    Y_cont_verif.index = np.arange(X.shape[0])
    return Y_cont_verif