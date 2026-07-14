from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
from numpy import np

def remove_pixels(image):
    """ 
    Replace NaNs and infinite values with zero.
    """
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)