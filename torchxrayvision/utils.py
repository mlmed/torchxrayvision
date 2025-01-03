import sys
import requests
import numpy as np
import skimage
import torch
import os

from os import PathLike
from numpy import ndarray
import warnings
from tqdm.autonotebook import tqdm


def get_cache_dir():
    return os.path.expanduser(os.path.join("~", ".torchxrayvision", "models_data/"))

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


def read_xray_dcm(path: PathLike, voi_lut: bool = False, fix_monochrome: bool = True) -> ndarray:
    """read a dicom-like file and convert to numpy array 

    Args:
        path (PathLike): path to the dicom file
        voi_lut (bool, optional): transform image to be human viewable. Defaults to False.
        fix_monochrome (bool, optional): Convert dicom interpretation MONOCHROME1 to MONOCHROME2. Defaults to True.

    Returns:
        ndarray: 2D single array image for a dicom image scaled between -1024, 1024
    """
    try:
        import pydicom
    except ImportError:
        raise Exception("Missing Package Pydicom. Try installing it by running `pip install pydicom`.")

    # get the pixel array
    ds = pydicom.dcmread(path, force=True)

    # we have not tested RGB, YBR_FULL, or YBR_FULL_422 yet.
    if ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2']:
        raise NotImplementedError(f'PhotometricInterpretation `{ds.PhotometricInterpretation}` is not yet supported.')
    # get the max possible pixel value from DCM header
    max_possible_pixel_val = (2**ds.BitsStored - 1)

    data = ds.pixel_array
    
    # LUT for human friendly view
    if voi_lut:
        data = pydicom.pixel_data_handlers.util.apply_voi_lut(data, ds, index=0)

    # `MONOCHROME1` have an inverted view; Bones are black; background is white
    # https://web.archive.org/web/20150920230923/http://www.mccauslandcenter.sc.edu/mricro/dicom/index.html
    if fix_monochrome and ds.PhotometricInterpretation == "MONOCHROME1":
        warnings.warn(f"Coverting MONOCHROME1 to MONOCHROME2 interpretation for file: {path}. Can be avoided by setting `fix_monochrome=False`")
        data = max_possible_pixel_val - data

    # normalize data to [-1024, 1024]
    data = normalize(data, max_possible_pixel_val)
    return data


def infer(model: torch.nn.Module, dataset: torch.utils.data.Dataset, threads=4, device='cpu'):

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=threads,
        num_workers=threads,
    )

    preds = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(dl)):
            output = model(batch["img"].to(device))

            output = output.detach().cpu().numpy()
            preds.append(output)

    return np.concatenate(preds)


warning_log = {}

def fix_resolution(x, resolution: int, model):
    """Check resolution of input and resize to match requested."""

    if len(x.shape) == 3:
        # Extend to be 4D
        x = x[None,...]

    if x.shape[2] != x.shape[3]:
        raise Exception(f"Height and width of the image must be the same. Input: {x.shape[2]} != {x.shape[3]}. Perform a center crop first.")
    
    if (x.shape[2] != resolution) | (x.shape[3] != resolution):
        if not hash(model) in warning_log:
            print("Warning: Input size ({}x{}) is not the native resolution ({}x{}) for this model. A resize will be performed but this could impact performance.".format(x.shape[2], x.shape[3], resolution, resolution))
            warning_log[hash(model)] = True
        return torch.nn.functional.interpolate(x, size=(resolution, resolution), mode='bilinear', antialias=True)
    return x


def warn_normalization(x):
    """Check normalization of input and warn if possibly wrong. When 
    processing an image that may likely not have the correct 
    normalization we can issue a warning. But running min and max on 
    every image/batch is costly so we only do it on the first image/batch.
    """

    # Only run this check on the first image so we don't hurt performance.
    if not "norm_check" in warning_log:
        x_min = x.min()
        x_max = x.max()
        if torch.logical_or(-255 < x_min, x_max < 255) or torch.logical_or(x_min < -1025, 1025 < x_max):
            print(f'Warning: Input image does not appear to be normalized correctly. The input image has the range [{x_min:.2f},{x_max:.2f}] which doesn\'t seem to be in the [-1024,1024] range. This warning may be wrong though. Only the first image is tested and we are only using a heuristic in an attempt to save a user from using the wrong normalization.')
            warning_log["norm_correct"] = False
        else:
            warning_log["norm_correct"] = True

        warning_log["norm_check"] = True
