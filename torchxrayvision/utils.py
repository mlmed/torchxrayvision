import sys
import requests
import numpy as np
import skimage
import pydicom
from typing import (
    BinaryIO, Union, Optional, List, Any, Callable, cast, MutableSequence,
    Iterator, Dict, Type
)
from pydicom.filebase import DicomFileLike
from pydicom.fileutil import PathType
from numpy import ndarray
import warnings


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


# from here https://sumit-ghosh.com/articles/python-download-progress-bar/
def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')


def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024]."""

    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))

    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :]

    return img


def load_image(fname: str):
    """Load an image from a file and normalize it between -1024 and 1024. Assumes 8-bits per pixel."""

    img = skimage.io.imread(fname)
    img = normalize(img, 255)

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]

    return img

def read_xray_dcm(path:Union[PathType, BinaryIO, DicomFileLike], voi_lut:bool=False, fix_monochrome:bool=True)->ndarray:
    """read a dicom-like file and convert to numpy array 

    Args:
        path (Union[PathType, BinaryIO, DicomFileLike]): path to the dicom file
        voi_lut (bool, optional): transform image to be human viewable. Defaults to False.
        fix_monochrome (bool, optional): Convert dicom interpretation MONOCHROME1 to MONOCHROME2. Defaults to True.

    Returns:
        ndarray: 2D single array image for a dicom image
    """

    dicom = pydicom.dcmread(path)
    
    # LUT for human friendly view
    if voi_lut:
        data = pydicom.pixel_data_handlers.util.apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # `MONOCHROME1` have an inverted view; Bones are black; background is white
    # https://web.archive.org/web/20150920230923/http://www.mccauslandcenter.sc.edu/mricro/dicom/index.html
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        warnings.warn(f"Coverting MONOCHROME1 to MONOCHROME2 inpretation for file: {path}. Can be avoided by setting `fix_monochrome=False`")
        data = np.amax(data) - data
        
    data = normalize(data, np.amax(data))
    
    return data