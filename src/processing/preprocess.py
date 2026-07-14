from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
import numpy as np

def remove_pixels(image):
    """ 
    Replace NaNs and infinite values with zero.
    """

    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) = find_hot_pixels(clean_image)