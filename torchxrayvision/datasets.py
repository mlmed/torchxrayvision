import collections
import os
import os.path
import pprint
import random
import sys
import tarfile
import warnings
import zipfile

import imageio
import numpy as np
import pandas as pd
import json
import skimage
from typing import Dict, List
from collections import defaultdict
import skimage.transform
from skimage.io import imread
import torch
from torchvision import transforms
import torchxrayvision as xrv

default_pathologies = [
    'Atelectasis',
    'Consolidation',
    'Infiltration',
    'Pneumothorax',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Effusion',
    'Pneumonia',
    'Pleural_Thickening',
    'Cardiomegaly',
    'Nodule',
    'Mass',
    'Hernia',
    'Lung Lesion',
    'Fracture',
    'Lung Opacity',
    'Enlarged Cardiomediastinum'
]

# Use a file that ships with the library
USE_INCLUDED_FILE = "USE_INCLUDED_FILE"

thispath = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(thispath, "data")

# this is for caching small things for speed
_cache_dict = {}


def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024].

    Call xrv.utils.normalize moving forward.
    """
    return xrv.utils.normalize(img, maxval, reshape)


def apply_transforms(sample, transform, seed=None) -> Dict:
    """Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or 
    other tasks which use masks information.
    """

    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])

        if "pathology_masks" in sample:
            for i in sample["pathology_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])

        if "semantic_masks" in sample:
            for i in sample["semantic_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])

    return sample


def relabel_dataset(pathologies, dataset, silent=False):
    """This function will add, remove, and reorder the `.labels` field to
have the same order as the pathologies argument passed to it. If a pathology is specified but doesn’t
exist in the dataset then a NaN will be put in place of the label.

    Args:
        :pathologies: The list of pathologies that the dataset will be aligned.
        :dataset: The dataset object that will be edited.
        :silent: Set True to silence printing details of the alignment.
    """
    will_drop = set(dataset.pathologies).difference(pathologies)
    if will_drop != set():
        if not silent:
            print("{} will be dropped".format(will_drop))
    new_labels = []
    dataset.pathologies = list(dataset.pathologies)
    for pathology in pathologies:
        if pathology in dataset.pathologies:
            pathology_idx = dataset.pathologies.index(pathology)
            new_labels.append(dataset.labels[:, pathology_idx])
        else:
            if not silent:
                print("{} doesn't exist. Adding nans instead.".format(pathology))
            values = np.empty(dataset.labels.shape[0])
            values.fill(np.nan)
            new_labels.append(values)
    new_labels = np.asarray(new_labels).T

    dataset.labels = new_labels
    dataset.pathologies = pathologies


class Dataset:
    """The datasets in this library aim to fit a simple interface where the
    imgpath and csvpath are specified. Some datasets require more than one
    metadata file and for some the metadata files are packaged in the library
    so only the imgpath needs to be specified.
    """

    def __init__(self):
        pass

    pathologies: List[str]
    """A list of strings identifying the pathologies contained in this 
    dataset. This list corresponds to the columns of the `.labels` matrix. 
    Although it is called pathologies, the contents do not have to be 
    pathologies and may simply be attributes of the patient. """

    labels: np.ndarray
    """A NumPy array which contains a 1, 0, or NaN for each pathology. Each 
    column is a pathology and each row corresponds to an item in the dataset. 
    A 1 represents that the pathology is present, 0 represents the pathology 
    is absent, and NaN represents no information. """

    csv: pd.DataFrame
    """A Pandas DataFrame of the metadata .csv file that is included with the 
    data. For some datasets multiple metadata files have been merged 
    together. It is largely a "catch-all" for associated data and the 
    referenced publication should explain each field. Each row aligns with 
    the elements of the dataset so indexing using .iloc will work. Alignment 
    between the DataFrame and the dataset items will be maintained when using 
    tools from this library. """

    def totals(self) -> Dict[str, Dict[str, int]]:
        """Compute counts of pathologies.

        Returns: A dict containing pathology name -> (label->value)
        """
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    def __repr__(self) -> str:
        """Returns the name and a description of the dataset such as:

        .. code-block:: python

            CheX_Dataset num_samples=191010 views=['PA', 'AP']

        If in a jupyter notebook it will also print the counts of the
        pathology counts returned by .totals()

        .. code-block:: python

            {'Atelectasis': {0.0: 17621, 1.0: 29718},
             'Cardiomegaly': {0.0: 22645, 1.0: 23384},
             'Consolidation': {0.0: 30463, 1.0: 12982},
             ...}

        """
        if xrv.utils.in_notebook():
            pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        self.csv["view"] = self.csv["view"].fillna("UNKNOWN")

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(self.views)]  # Select the view


class MergeDataset(Dataset):
    """The class `MergeDataset` can be used to merge multiple datasets
    together into a single dataset. This class takes in a list of dataset
    objects and assembles the datasets in order. This class will correctly
    maintain the `.labels`, `.csv`, and `.pathologies` fields and offer
    pretty printing.

    .. code-block:: python

        dmerge = xrv.datasets.MergeDataset([dataset1, dataset2, ...])
        # Output:
        MergeDataset num_samples=261583
        - 0 PC_Dataset num_samples=94825 views=['PA', 'AP']
        - 1 RSNA_Pneumonia_Dataset num_samples=26684 views=['PA', 'AP']
        - 2 NIH_Dataset num_samples=112120 views=['PA', 'AP']
        - 3 SIIM_Pneumothorax_Dataset num_samples=12954
        - 4 VinBrain_Dataset num_samples=15000 views=['PA', 'AP']
    """

    def __init__(self, datasets, seed=0, label_concat=False):
        super(MergeDataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.datasets = datasets
        self.length = 0
        self.pathologies = datasets[0].pathologies
        self.which_dataset = np.zeros(0)
        self.offset = np.zeros(0)
        currentoffset = 0
        for i, dataset in enumerate(datasets):
            self.which_dataset = np.concatenate([self.which_dataset, np.zeros(len(dataset)) + i])
            self.length += len(dataset)
            self.offset = np.concatenate([self.offset, np.zeros(len(dataset)) + currentoffset])
            currentoffset += len(dataset)
            if dataset.pathologies != self.pathologies:
                raise Exception("incorrect pathology alignment")

        if hasattr(datasets[0], 'labels'):
            self.labels = np.concatenate([d.labels for d in datasets])
        else:
            print("WARN: not adding .labels")

        self.which_dataset = self.which_dataset.astype(int)

        if label_concat:
            new_labels = np.zeros([self.labels.shape[0], self.labels.shape[1] * len(datasets)]) * np.nan
            for i, shift in enumerate(self.which_dataset):
                size = self.labels.shape[1]
                new_labels[i, shift * size:shift * size + size] = self.labels[i]
            self.labels = new_labels

        try:
            self.csv = pd.concat([d.csv for d in datasets])
        except:
            print("Could not merge dataframes (.csv not available):", sys.exc_info()[0])

        self.csv = self.csv.reset_index(drop=True)

    def __setattr__(self, name, value):
        if hasattr(self, 'labels'):
            # check only if have finished init, otherwise __init__ breaks
            if name in ['transform', 'data_aug', 'labels', 'pathologies', 'targets']:
                raise NotImplementedError(f'Cannot set {name} on a merged dataset. Set the transforms directly on the dataset object. If it was to be set via this merged dataset it would have to modify the internal dataset which could have unexpected side effects')

        object.__setattr__(self, name, value)

    def string(self):
        s = self.__class__.__name__ + " num_samples={}\n".format(len(self))
        for i, d in enumerate(self.datasets):
            if i < len(self.datasets) - 1:
                s += "├{} ".format(i) + d.string().replace("\n", "\n|  ") + "\n"
            else:
                s += "└{} ".format(i) + d.string().replace("\n", "\n   ") + "\n"
        return s

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.datasets[int(self.which_dataset[idx])][idx - int(self.offset[idx])]
        item["lab"] = self.labels[idx]
        item["source"] = self.which_dataset[idx]
        return item


# alias so it is backwards compatible
Merge_Dataset = MergeDataset


class FilterDataset(Dataset):
    def __init__(self, dataset, labels=None):
        super(FilterDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        self.idxs = []
        if labels:
            for label in labels:
                print("filtering for ", label)

                self.idxs += list(np.where(dataset.labels[:, list(dataset.pathologies).index(label)] == 1)[0])

        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]

    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]


class SubsetDataset(Dataset):
    """When you only want a subset of a dataset the `SubsetDataset` class can
    be used. A list of indexes can be passed in and only those indexes will
    be present in the new dataset. This class will correctly maintain the
    `.labels`, `.csv`, and `.pathologies` fields and offer pretty printing.

    .. code-block:: python

        dsubset = xrv.datasets.SubsetDataset(dataset, [0, 5, 60])
        # Output:
        SubsetDataset num_samples=3
        of PC_Dataset num_samples=94825 views=['PA', 'AP']

    For example this class can be used to create a dataset of only female
    patients by selecting that column of the csv file and using np.where to
    convert this boolean vector into a list of indexes.

    .. code-block:: python

        idxs = np.where(dataset.csv.PatientSex_DICOM=="F")[0]
        dsubset = xrv.datasets.SubsetDataset(dataset, idxs)
        # Output:
        SubsetDataset num_samples=48308
        - of PC_Dataset num_samples=94825 views=['PA', 'AP'] data_aug=None

    """

    def __init__(self, dataset, idxs=None):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        self.idxs = idxs
        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]
        self.csv = self.csv.reset_index(drop=True)

        if hasattr(self.dataset, 'which_dataset'):
            # keep information about the source dataset from a merged dataset
            self.which_dataset = self.dataset.which_dataset[self.idxs]

    def __setattr__(self, name, value):
        if hasattr(self, 'labels'):
            # check only if have finished init, otherwise __init__ breaks
            if name in ['transform', 'data_aug', 'labels', 'pathologies', 'targets']:
                raise NotImplementedError(f'Cannot set {name} on a subset dataset. Set the transforms directly on the dataset object. If it was to be set via this subset dataset it would have to modify the internal dataset which could have unexpected side effects')

        object.__setattr__(self, name, value)

    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]


class NIH_Dataset(Dataset):
    """NIH ChestX-ray14 dataset

    The NIH ChestX-ray14 dataset contains 112,120 frontal-view chest X-ray
    images from 30,805 unique patients. Each image may carry one or more of
    14 disease labels that were automatically mined from accompanying
    radiological reports using natural language processing. The text-mined
    labels are expected to have an accuracy greater than 90 %.

    **Pathologies (14):** Atelectasis, Cardiomegaly, Consolidation, Edema,
    Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule,
    Pleural Thickening, Pneumonia, Pneumothorax.

    Bounding-box annotations for a subset of images are included and are
    accessible via the ``pathology_masks=True`` argument.

    Citation:
        Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM.
        ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.
        *Proceedings of CVPR*, 2017.
        https://arxiv.org/abs/1705.02315

    Dataset release:
        https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

    Download full-size images:
        https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

    Download resized (224 × 224) images:
        https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 bbox_list_path=USE_INCLUDED_FILE,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 unique_patients=True,
                 pathology_masks=False
                 ):
        super(NIH_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "Data_Entry_2017_v2020.csv.gz")
        else:
            self.csvpath = csvpath

        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath)

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()

        self.csv = self.csv.reset_index()

        ####### pathology masks ########
        # load nih pathology masks
        if bbox_list_path == USE_INCLUDED_FILE:
            self.bbox_list_path = os.path.join(datapath, "BBox_List_2017.csv.gz")
        else:
            self.bbox_list_path = bbox_list_path
        self.pathology_maskscsv = pd.read_csv(
            self.bbox_list_path,
            names=["Image Index", "Finding Label", "x", "y", "w", "h", "_1", "_2", "_3"],
            skiprows=1
        )

        # change label name to match
        self.pathology_maskscsv.loc[self.pathology_maskscsv["Finding Label"] == "Infiltrate", "Finding Label"] = "Infiltration"
        self.csv["has_masks"] = self.csv["Image Index"].isin(self.pathology_maskscsv["Image Index"])

        ####### pathology masks ########
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # add consistent csv values

        # offset_day_int
        # self.csv["offset_day_int"] =

        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)

        # age
        self.csv['age_years'] = self.csv['Patient Age'] * 1.0

        # sex
        self.csv['sex_male'] = self.csv['Patient Gender'] == 'M'
        self.csv['sex_female'] = self.csv['Patient Gender'] == 'F'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_mask_dict(self, image_name, this_size):
        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.pathology_maskscsv[self.pathology_maskscsv["Image Index"] == image_name]
        path_mask = {}

        for i in range(len(images_with_masks)):
            row = images_with_masks.iloc[i]

            # Don't add masks for labels we don't have
            if row["Finding Label"] in self.pathologies:
                mask = np.zeros([this_size, this_size])
                xywh = np.asarray([row.x, row.y, row.w, row.h])
                xywh = xywh * scale
                xywh = xywh.astype(int)
                mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

                # Resize so image resizing works
                mask = mask[None, :, :]

                path_mask[self.pathologies.index(row["Finding Label"])] = mask
        return path_mask


class RSNA_Pneumonia_Dataset(Dataset):
    """RSNA Pneumonia Detection Challenge dataset

    A subset of the NIH ChestX-ray14 images re-annotated by board-certified
    radiologists for the 2018 RSNA Pneumonia Detection Challenge. The dataset
    contains 26,684 frontal chest X-rays with bounding-box annotations for
    regions of pneumonia / lung opacity.

    **Pathologies (2):** Lung Opacity, Pneumonia.

    Per-image bounding-box masks are available via ``pathology_masks=True``.
    Images can be loaded as JPEG (default) or DICOM by setting
    ``extension=".dcm"`` (requires ``pydicom``).

    Citation:
        Shih G, Wu CC, Halabi SS, et al.
        Augmenting the National Institutes of Health Chest Radiograph Dataset
        with Expert Annotations of Possible Pneumonia.
        *Radiology: Artificial Intelligence*, 2019.
        doi: 10.1148/ryai.2019180041

    Challenge site:
        https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

    Download JPEG images:
        https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 dicomcsvpath=USE_INCLUDED_FILE,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 pathology_masks=False,
                 extension=".jpg"
                 ):

        super(RSNA_Pneumonia_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.pathologies = ["Pneumonia", "Lung Opacity"]

        self.pathologies = sorted(self.pathologies)

        self.extension = extension
        self.use_pydicom = (extension == ".dcm")

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "kaggle_stage_2_train_labels.csv.zip")
        else:
            self.csvpath = csvpath
        self.raw_csv = pd.read_csv(self.csvpath, nrows=nrows)

        # The labels have multiple instances for each mask
        # So we just need one to get the target label
        self.csv = self.raw_csv.groupby("patientId").first()

        if dicomcsvpath == USE_INCLUDED_FILE:
            self.dicomcsvpath = os.path.join(datapath, "kaggle_stage_2_train_images_dicom_headers.csv.gz")
        else:
            self.dicomcsvpath = dicomcsvpath

        self.dicomcsv = pd.read_csv(self.dicomcsvpath, nrows=nrows, index_col="PatientID")

        self.csv = self.csv.join(self.dicomcsv, on="patientId")

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['ViewPosition']
        self.limit_to_selected_views(views)

        self.csv = self.csv.reset_index()

        # Get our classes.
        labels = [self.csv["Target"].values, self.csv["Target"].values]

        # set if we have masks
        self.csv["has_masks"] = ~np.isnan(self.csv["x"])

        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # add consistent csv values

        # offset_day_int
        # TODO: merge with NIH metadata to get dates for images

        # patientid
        self.csv["patientid"] = self.csv["patientId"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['patientId'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid + self.extension)

        if self.use_pydicom:
            try:
                import pydicom
            except ImportError as e:
                raise Exception("Please install pydicom to work with this dataset")

            img = pydicom.filereader.dcmread(img_path).pixel_array
        else:
            img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_mask_dict(self, image_name, this_size):

        base_size = 1024
        scale = this_size / base_size

        images_with_masks = self.raw_csv[self.raw_csv["patientId"] == image_name]
        path_mask = {}

        # All masks are for both pathologies
        for patho in ["Pneumonia", "Lung Opacity"]:
            mask = np.zeros([this_size, this_size])

            # Don't add masks for labels we don't have
            if patho in self.pathologies:

                for i in range(len(images_with_masks)):
                    row = images_with_masks.iloc[i]
                    xywh = np.asarray([row.x, row.y, row.width, row.height])
                    xywh = xywh * scale
                    xywh = xywh.astype(int)
                    mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

            # Resize so image resizing works
            mask = mask[None, :, :]

            path_mask[self.pathologies.index(patho)] = mask
        return path_mask


class NIH_Google_Dataset(Dataset):
    """NIH ChestX-ray14 with Google radiologist re-labels

    A subset of the NIH ChestX-ray14 dataset that has been re-annotated by
    radiologists at Google. Labels were adjudicated by a panel of US
    board-certified radiologists to produce high-quality reference standards
    for four findings.

    **Pathologies (4):** Airspace Opacity (mapped to Lung Opacity), Fracture,
    Nodule/Mass, Pneumothorax.

    The original release provides separate test and validation splits; this
    class combines both by default. To use only one split, pass the
    corresponding CSV file via the ``csvpath`` argument.

    .. note::
        This class loads images from an existing NIH ChestX-ray14 download.
        The image files themselves are not redistributed.

    Citation:
        Majkowska A, Mittal S, Steiner DF, et al.
        Chest Radiograph Interpretation with Deep Learning Models: Assessment
        with Radiologist-adjudicated Reference Standards and
        Population-adjusted Evaluation.
        *Radiology*, 2020.
        https://pubs.rsna.org/doi/10.1148/radiol.2019191293

    Download NIH images (resized 224 × 224):
        https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True
                 ):

        super(NIH_Google_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        self.pathologies = ["Fracture", "Pneumothorax", "Airspace opacity",
                            "Nodule or mass"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "google2019_nih-chest-xray-labels.csv.gz")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first().reset_index()

        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv[pathology] == "YES"

            self.labels.append(mask.values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # rename pathologies
        self.pathologies = np.char.replace(self.pathologies, "Airspace opacity", "Lung Opacity")
        self.pathologies = np.char.replace(self.pathologies, "Nodule or mass", "Nodule/Mass")
        self.pathologies = list(self.pathologies)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class PC_Dataset(Dataset):
    """PadChest dataset

    A large, multi-label chest X-ray dataset collected at the Hospital
    San Juan de Alicante (Spain). PadChest contains over 160,000 images
    from more than 67,000 patients, annotated with 174 radiographic findings
    across 27 diagnostic labels (28 as loaded here, including a support
    devices label). Labels were obtained via a combination of manual
    annotation and NLP applied to Spanish-language radiology reports.
    Roughly a quarter of the images were manually verified by a radiologist.

    **Pathologies (28):** Atelectasis, Cardiomegaly, Consolidation, and
    many more — see ``self.pathologies`` for the full list after loading.

    .. note::
        Images with null labels (distinct from a normal finding) and a small
        number of images that cannot be loaded are excluded at load time, so
        the effective dataset size is slightly less than the file count.

    Citation:
        Bustos A, Pertusa A, Salinas JM, de la Iglesia-Vayá M.
        PadChest: A large chest x-ray image dataset with multi-label
        annotated reports.
        *arXiv:1901.07441*, 2019.
        https://arxiv.org/abs/1901.07441

    Dataset website:
        http://bimcv.cipf.es/bimcv-projects/padchest/

    Download full-size images:
        https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850

    Download resized (224 × 224) images:
        https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=True
                 ):

        super(PC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia", "Fracture",
                            "Granuloma", "Flattened Diaphragm", "Bronchiectasis",
                            "Aortic Elongation", "Scoliosis",
                            "Hilar Enlargement", "Tuberculosis",
                            "Air Trapping", "Costophrenic Angle Blunting", "Aortic Atheromatosis",
                            "Hemidiaphragm Elevation",
                            "Support Devices", "Tube'"]  # the Tube' is intentional

        self.pathologies = sorted(self.pathologies)

        mapping = dict()

        mapping["Infiltration"] = ["infiltrates",
                                   "interstitial pattern",
                                   "ground glass pattern",
                                   "reticular interstitial pattern",
                                   "reticulonodular interstitial pattern",
                                   "alveolar pattern",
                                   "consolidation",
                                   "air bronchogram"]
        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Consolidation"] = ["air bronchogram"]
        mapping["Hilar Enlargement"] = ["adenopathy",
                                        "pulmonary artery enlargement"]
        mapping["Support Devices"] = ["device",
                                      "pacemaker"]
        mapping["Tube'"] = ["stent'"]  # the ' is to select findings which end in that word

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.flat_dir = flat_dir
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz")
        else:
            self.csvpath = csvpath

        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, low_memory=False)

        # Standardize view names
        self.csv.loc[self.csv["Projection"].isin(["AP_horizontal"]), "Projection"] = "AP Supine"

        self.csv["view"] = self.csv['Projection']
        self.limit_to_selected_views(views)

        # Remove null stuff
        self.csv = self.csv[~self.csv["Labels"].isnull()]

        # Remove missing files
        missing = ["216840111366964012819207061112010307142602253_04-014-084.png",
                   "216840111366964012989926673512011074122523403_00-163-058.png",
                   "216840111366964012959786098432011033083840143_00-176-115.png",
                   "216840111366964012558082906712009327122220177_00-102-064.png",
                   "216840111366964012339356563862009072111404053_00-043-192.png",
                   "216840111366964013076187734852011291090445391_00-196-188.png",
                   "216840111366964012373310883942009117084022290_00-064-025.png",
                   "216840111366964012283393834152009033102258826_00-059-087.png",
                   "216840111366964012373310883942009170084120009_00-097-074.png",
                   "216840111366964012819207061112010315104455352_04-024-184.png",
                   "216840111366964012819207061112010306085429121_04-020-102.png",
                   "216840111366964012989926673512011083134050913_00-168-009.png",  # broken PNG file (chunk b'\x00\x00\x00\x00')
                   "216840111366964012373310883942009152114636712_00-102-045.png",  # "OSError: image file is truncated"
                   "216840111366964012819207061112010281134410801_00-129-131.png",  # "OSError: image file is truncated"
                   "216840111366964012487858717522009280135853083_00-075-001.png",  # "OSError: image file is truncated"
                   "216840111366964012989926673512011151082430686_00-157-045.png",  # broken PNG file (chunk b'\x00\x00\x00\x00')
                   "216840111366964013686042548532013208193054515_02-026-007.png",  # "OSError: image file is truncated"
                   "216840111366964013590140476722013058110301622_02-056-111.png",  # "OSError: image file is truncated"
                   "216840111366964013590140476722013043111952381_02-065-198.png",  # "OSError: image file is truncated"
                   "216840111366964013829543166512013353113303615_02-092-190.png",  # "OSError: image file is truncated"
                   "216840111366964013962490064942014134093945580_01-178-104.png",  # "OSError: image file is truncated"
                   ]
        self.csv = self.csv[~self.csv["ImageID"].isin(missing)]

        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        self.csv = self.csv.sort_values("ImageID").reset_index(drop=True)

        # Filter out age < 10 (paper published 2019)
        self.csv = self.csv[(2019 - self.csv.PatientBirth > 10)]

        # Get our classes.
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    # print("mapping", syn)
                    mask |= self.csv["Labels"].str.contains(syn.lower())
            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.pathologies[self.pathologies.index("Tube'")] = "Tube"

        # add consistent csv values

        # offset_day_int
        dt = pd.to_datetime(self.csv["StudyDate_DICOM"], format="%Y%m%d")
        self.csv["offset_day_int"] = dt.astype(np.int64) // 10**9 // 86400

        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)

        # age
        self.csv['age_years'] = (2017 - self.csv['PatientBirth'])

        # sex
        self.csv['sex_male'] = self.csv['PatientSex_DICOM'] == 'M'
        self.csv['sex_female'] = self.csv['PatientSex_DICOM'] == 'F'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['ImageID'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=65535, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class CheX_Dataset(Dataset):
    """CheXpert dataset (Stanford)

    CheXpert is a large chest radiograph dataset from Stanford containing
    224,316 images from 65,240 patients. Labels for 14 observations were
    generated automatically using the CheXpert labeler applied to free-text
    radiology reports. A key feature of this dataset is its handling of
    *uncertain* labels: the original CSV encodes uncertainty as ``-1``, which
    this class converts to ``NaN`` for consistency with the rest of the
    library.

    **Pathologies (13):** Atelectasis, Cardiomegaly, Consolidation, Edema,
    Effusion, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung
    Opacity, Pleural Other, Pneumonia, Pneumothorax, Support Devices.
    ("No Finding" is used internally to zero-out pathology labels but is not
    returned as a column.)

    Citation:
        Irvin J, Rajpurkar P, Ko M, et al.
        CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels
        and Expert Comparison.
        *arXiv:1901.07031*, 2019.
        https://arxiv.org/abs/1901.07031

    Dataset website:
        https://stanfordmlgroup.github.io/competitions/chexpert/
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=True
                 ):

        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "chexpert_train.csv.gz")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views

        self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"]  # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace({'Lateral': "L"})  # Rename Lateral with L

        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat=r'(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
            else:
                mask = pd.Series(np.nan, index=self.csv.index)

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        # Rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))

        # add consistent csv values

        # offset_day_int

        # patientid
        if 'train' in self.csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif 'valid' in self.csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplementedError

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient", "")

        # patientid
        self.csv["patientid"] = patientid

        # age
        self.csv['age_years'] = self.csv['Age'] * 1.0
        self.csv.loc[self.csv['Age'] == 0, 'Age'] = None

        # sex
        self.csv['sex_male'] = self.csv['Sex'] == 'Male'
        self.csv['sex_female'] = self.csv['Sex'] == 'Female'

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Path'].iloc[idx]
        # clean up path in csv so the user can specify the path
        imgid = imgid.replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0/", "")
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class MIMIC_Dataset(Dataset):
    """MIMIC-CXR dataset (MIT / Beth Israel Deaconess Medical Center)

    MIMIC-CXR is a large, de-identified dataset of chest radiographs
    collected from the Beth Israel Deaconess Medical Center between 2011 and
    2016. It contains 227,835 images from 64,588 patients, along with
    structured labels extracted from free-text radiology reports using the
    CheXpert labeler. Both PA and AP views are available.

    **Pathologies (13):** Atelectasis, Cardiomegaly, Consolidation, Edema,
    Effusion, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung
    Opacity, Pleural Other, Pneumonia, Pneumothorax, Support Devices.

    .. note::
        Access requires a credentialed PhysioNet account and completion of
        the required data-use training. Both a ``csvpath`` (labels CSV) and
        a ``metacsvpath`` (DICOM metadata CSV) must be provided.

    Citation:
        Johnson AEW, Pollard TJ, Berkowitz S, et al.
        MIMIC-CXR: A large publicly available database of labeled chest
        radiographs.
        *arXiv:1901.07042*, 2019.
        https://arxiv.org/abs/1901.07042

    Dataset website:
        https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self,
                 imgpath,
                 csvpath,
                 metacsvpath,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 unique_patients=True
                 ):

        super(MIMIC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]

        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.metacsv = pd.read_csv(self.metacsvpath)

        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        self.metacsv = self.metacsv.set_index(['subject_id', 'study_id'])

        self.csv = self.csv.join(self.metacsv).reset_index()

        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
            else:
                mask = pd.Series(np.nan, index=self.csv.index)

            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan

        # Rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))

        # add consistent csv values

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]

        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])

        img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class Openi_Dataset(Dataset):
    """OpenI / Indiana University chest X-ray collection

    The Indiana University chest X-ray collection (OpenI) contains 7,470
    chest X-ray images from 3,955 radiology reports collected at Indiana
    University Health. Labels are derived automatically from MeSH terms
    embedded in the XML report files.

    **Pathologies (18):** Atelectasis, Calcified Granuloma, Cardiomegaly,
    Edema, Effusion, Emphysema, Fibrosis, Fracture, Granuloma, Hernia,
    Infiltration, Lung Lesion, Lung Opacity, Mass, Nodule, Pleural
    Thickening, Pneumonia, Pneumothorax.

    .. note::
        View position labels in the original records are noisy. A T-SNE
        projection was used to derive higher-quality PA/AP labels. 
        Set ``use_tsne_derived_view=True`` to use these
        derived labels instead of the raw metadata values.

    Citation:
        Demner-Fushman D, Kohli MD, Rosenman MB, et al.
        Preparing a collection of radiology examinations for distribution
        and retrieval.
        *Journal of the American Medical Informatics Association*, 2016.
        doi: 10.1093/jamia/ocv080

    Dataset website:
        https://openi.nlm.nih.gov/faq

    Download images:
        https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d
    """

    def __init__(self, imgpath,
                 xmlpath=USE_INCLUDED_FILE,
                 dicomcsv_path=USE_INCLUDED_FILE,
                 tsnepacsv_path=USE_INCLUDED_FILE,
                 use_tsne_derived_view=False,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True
                 ):

        super(Openi_Dataset, self).__init__()
        import xml
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        self.pathologies = ["Atelectasis", "Fibrosis",
                            "Pneumonia", "Effusion", "Lesion",
                            "Cardiomegaly", "Calcified Granuloma",
                            "Fracture", "Edema", "Granuloma", "Emphysema",
                            "Hernia", "Mass", "Nodule", "Opacity", "Infiltration",
                            "Pleural_Thickening", "Pneumothorax", ]

        self.pathologies = sorted(self.pathologies)

        mapping = dict()

        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Infiltration"] = ["Infiltrate"]
        mapping["Atelectasis"] = ["Atelectases"]

        # Load data
        if xmlpath == USE_INCLUDED_FILE:
            self.xmlpath = os.path.join(datapath, "NLMCXR_reports.tgz")
        else:
            self.xmlpath = xmlpath

        tarf = tarfile.open(self.xmlpath, 'r:gz')

        samples = []

        for filename in tarf.getnames():
            if (filename.endswith(".xml")):
                tree = xml.etree.ElementTree.parse(tarf.extractfile(filename))
                root = tree.getroot()
                uid = root.find("uId").attrib["id"]
                labels_m = [node.text.lower() for node in root.findall(".//MeSH/major")]
                labels_m = "|".join(np.unique(labels_m))
                labels_a = [node.text.lower() for node in root.findall(".//MeSH/automatic")]
                labels_a = "|".join(np.unique(labels_a))
                image_nodes = root.findall(".//parentImage")
                for image in image_nodes:
                    sample = {}
                    sample["uid"] = uid
                    sample["imageid"] = image.attrib["id"]
                    sample["labels_major"] = labels_m
                    sample["labels_automatic"] = labels_a
                    samples.append(sample)

        self.csv = pd.DataFrame(samples)

        if dicomcsv_path == USE_INCLUDED_FILE:
            self.dicomcsv_path = os.path.join(datapath, "nlmcxr_dicom_metadata.csv.gz")
        else:
            self.dicomcsv_path = dicomcsv_path

        self.dicom_metadata = pd.read_csv(self.dicomcsv_path, index_col="imageid", low_memory=False)

        # Merge in dicom metadata
        self.csv = self.csv.join(self.dicom_metadata, on="imageid")

        if tsnepacsv_path == USE_INCLUDED_FILE:
            self.tsnepacsv_path = os.path.join(datapath, "nlmcxr_tsne_pa.csv.gz")
        else:
            self.tsnepacsv_path = tsnepacsv_path

        # Attach view computed by tsne
        tsne_pa = pd.read_csv(self.tsnepacsv_path, index_col="imageid")
        self.csv = self.csv.join(tsne_pa, on="imageid")

        if use_tsne_derived_view:
            self.csv["view"] = self.csv["tsne-view"]
        else:
            self.csv["view"] = self.csv["View Position"]

        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("uid").first().reset_index()

        # Get our classes.
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["labels_automatic"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    # print("mapping", syn)
                    mask |= self.csv["labels_automatic"].str.contains(syn.lower())
            labels.append(mask.values)

        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # Rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Opacity", "Lung Opacity"))
        self.pathologies = list(np.char.replace(self.pathologies, "Lesion", "Lung Lesion"))

        # add consistent csv values

        # offset_day_int
        # self.csv["offset_day_int"] =

        # patientid
        self.csv["patientid"] = self.csv["uid"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imageid = self.csv.iloc[idx].imageid
        img_path = os.path.join(self.imgpath, imageid + ".png")
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class COVID19_Dataset(Dataset):
    """COVID-19 Image Data Collection

    A manually curated, open-source collection of frontal and lateral chest
    X-rays (and CT scans) from COVID-19 cases, aggregated from published
    figures and public web repositories. It is one of the largest public
    resources for COVID-19 chest imaging and prognostic data.

    In addition to image labels, the accompanying metadata CSV provides
    clinical context including time since first symptoms, ICU admission
    status, survival status, intubation status, and hospital location.
    These fields enable tasks such as severity prediction and patient
    trajectory modelling.

    Lung segmentation masks (from V7 Labs) are optionally available via
    ``semantic_masks=True``.

    .. note::
        Both ``imgpath`` and ``csvpath`` must be provided; neither is bundled
        with the library. Clone or download the dataset repository first.

    Citations:
        Cohen JP, Morrison P, Dao L, Roth K, Duong TQ, Ghassemi M.
        COVID-19 Image Data Collection: Prospective Predictions Are the
        Future. *arXiv:2006.11988*, 2020.

        Cohen JP, Morrison P, Dao L.
        COVID-19 Image Data Collection.
        *arXiv:2003.11597*, 2020.

    Dataset repository:
        https://github.com/ieee8023/covid-chestxray-dataset
    """

    dataset_url = "https://github.com/ieee8023/covid-chestxray-dataset"

    def __init__(self,
                 imgpath: str,
                 csvpath: str,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 seed: int = 0,
                 semantic_masks=False,
                 ):
        super(COVID19_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.views = views
        self.semantic_masks = semantic_masks
        self.semantic_masks_v7labs_lungs_path = os.path.join(datapath, "semantic_masks_v7labs_lungs.zip")

        if not os.path.exists(csvpath):
            raise FileNotFoundError(f'The csvpath does not point to a valid metadata.csv file. Please download it from {self.dataset_url}')

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)

        # Keep only the selected views.
        self.limit_to_selected_views(views)

        # Filter out in progress samples
        self.csv = self.csv[~(self.csv.finding == "todo")]
        self.csv = self.csv[~(self.csv.finding == "Unknown")]

        self.pathologies = self.csv.finding.str.split("/", expand=True).values.ravel()
        self.pathologies = self.pathologies[~pd.isnull(self.pathologies)]
        self.pathologies = sorted(np.unique(self.pathologies))

        labels = []
        for pathology in self.pathologies:
            mask = self.csv["finding"].str.contains(pathology)
            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.csv = self.csv.reset_index()

        if self.semantic_masks:
            temp = zipfile.ZipFile(self.semantic_masks_v7labs_lungs_path)
            self.semantic_masks_v7labs_lungs_namelist = temp.namelist()

        # add consistent csv values

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["offset"]

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['filename'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.semantic_masks:
            sample["semantic_masks"] = self.get_semantic_mask_dict(imgid, sample["img"].shape)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_semantic_mask_dict(self, image_name, this_shape):

        archive_path = "semantic_masks_v7labs_lungs/" + image_name
        semantic_masks = {}
        if archive_path in self.semantic_masks_v7labs_lungs_namelist:
            with zipfile.ZipFile(self.semantic_masks_v7labs_lungs_path).open(archive_path) as file:
                mask = imageio.imread(file.read())

                mask = (mask == 255).astype(np.float32)
                # Reshape so image resizing works
                mask = mask[None, :, :]

                semantic_masks["Lungs"] = mask

        return semantic_masks


class NLMTB_Dataset(Dataset):
    """NLM Tuberculosis datasets (Montgomery County & Shenzhen)

    Two public chest X-ray datasets released by the National Library of
    Medicine for computer-aided TB screening:

    * **Montgomery County** (USA): 138 normal and 80 TB-positive PA images,
      collected by the Montgomery County Department of Health and Human
      Services.
    * **Shenzhen** (China): approximately 326 normal and 336 TB-positive PA
      images, collected at Shenzhen No. 3 People's Hospital.

    **Pathologies (1):** Tuberculosis.

    .. note::
        Load each dataset separately by pointing ``imgpath`` at the
        corresponding root folder (``NLM-MontgomeryCXRSet`` or
        ``ChinaSet_AllFiles``). Use
        :class:`~torchxrayvision.datasets.MergeDataset` to combine them.
        All images are PA view.

    Citation:
        Jaeger S, Candemir S, Antani S, Wang YX, Lu PX, Thoma G.
        Two public chest X-ray datasets for computer-aided screening of
        pulmonary diseases.
        *Quant Imaging Med Surg*, 2014; 4(6):475–477.
        doi: 10.3978/j.issn.2223-4292.2014.11.20

    Download Montgomery County images:
        https://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33

    Download Shenzhen images:
        https://academictorrents.com/details/462728e890bd37c05e9439c885df7afc36209cc8
    """

    def __init__(self,
                 imgpath,
                 transform=None,
                 data_aug=None,
                 seed=0,
                 views=["PA"]
                 ):
        """
        Args:
            img_path (str): Path to `MontgomerySet` or `ChinaSet_AllFiles`
                folder
        """

        super(NLMTB_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        file_list = []
        source_list = []

        for fname in sorted(os.listdir(os.path.join(self.imgpath, "CXR_png"))):
            if fname.endswith(".png"):
                file_list.append(fname)

        self.csv = pd.DataFrame({"fname": file_list})

        # Label is the last digit on the simage filename
        self.csv["label"] = self.csv["fname"].apply(lambda x: int(x.split(".")[-2][-1]))
        # All the images are PA according to the article.
        self.csv["view"] = "PA"
        self.limit_to_selected_views(views)

        self.labels = self.csv["label"].values.reshape(-1, 1)
        self.pathologies = ["Tuberculosis"]

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        item = self.csv.iloc[idx]
        img_path = os.path.join(self.imgpath, "CXR_png", item["fname"])
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

class TBX11K_Dataset(Dataset):
    """TBX11K Tuberculosis X-ray dataset

    TBX11K contains 11,200 chest X-ray images with bounding-box annotations
    for tuberculosis (TB) areas, spanning five categories: Healthy, Sick but
    Non-TB, Active TB, Latent TB, and Uncertain TB.

    .. note::
        This dataset overlaps with :class:`~xrv.datasets.NLMTB_Dataset`
        (Montgomery + Shenzhen images). Avoid training on one and evaluating
        on the other to prevent data leakage.    

    **Pathologies (4):** ActiveTuberculosis, ObsoletePulmonaryTuberculosis,
    PulmonaryTuberculosis, Tuberculosis (superclass).

    Label notes:

    - ``ActiveTuberculosis``: currently active, contagious TB, typically
      shown by infiltrates, consolidation, or cavities on the X-ray.
    - ``ObsoletePulmonaryTuberculosis``: old, healed/inactive TB lesions
      from a prior infection, no longer active.
    - ``PulmonaryTuberculosis``: a general pulmonary TB category defined in
      the dataset. Does not appear in train/val/trainval annotations so its
      label column is always 0.
    - ``Tuberculosis``: superclass label, positive if any of the above TB
      findings are present. Use this column for binary TB vs. non-TB tasks.

    This dataset incorporates images from four TB collections:

    - DA dataset (156 images, CC BY 4.0)
    - DB dataset (150 images, CC BY 4.0)
    - Montgomery County X-ray Set (138 images, public domain, NLM)
    - Shenzhen X-ray Set (662 images, public domain, NLM)

    Citation:
        Liu Y, Wu YH, Ban Y, Wang H, Cheng MM.
        Rethinking Computer-Aided Tuberculosis Diagnosis.
        *IEEE/CVF CVPR*, 2020, pp. 2643–2652.
        doi: 10.1109/CVPR42600.2020.00272

    Dataset and annotations:
        https://github.com/yun-liu/Tuberculosis

    Paper:
        https://ieeexplore.ieee.org/document/9156613

    License:
        CC BY-NC-SA 2.0 — https://creativecommons.org/licenses/by-nc-sa/2.0/
    """

    def __init__(self,
                imgpath,
                split="train",
                transform=None,
                data_aug=None,
                seed=0
                ):
        split_to_json = {
            "train": "TBX11K_train.json",
            "val": "TBX11K_val.json",
            "trainval": "TBX11K_trainval.json",
        }

        super(TBX11K_Dataset, self).__init__()
        np.random.seed(seed)
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        if split not in split_to_json:
            raise ValueError(f"Split must be one of {list(split_to_json.keys())}, got '{split}'")

        with open(os.path.join(self.imgpath, "annotations", "json", split_to_json[split])) as f:
            data = json.load(f)
        
        self.csv = pd.DataFrame(data["images"])
        ann_dict = defaultdict(list)
        for ann in data["annotations"]:
            ann_dict[ann["image_id"]].append({"category_id" : ann["category_id"], "bbox" : ann["bbox"]})
        self.csv['bbox'] = self.csv['id'].map(lambda x: [a["bbox"] for a in ann_dict[x]])

        # Create pathologies list by pulling every category dictionary and extracting the "name" value.
        # The pathology names define the label columns and their order
        self.pathologies = [cat["name"] for cat in data["categories"]]
        self.pathologies.append("Tuberculosis")
        # Map each category name to its numeric ID from the JSON
        cat_map = {cat["name"]: cat["id"] for cat in data["categories"]}
        # For each pathology, mark an image positive (1.0) if ANY of its annotations
        # match that category's ID. Using any() handles images with multiple bounding
        # boxes, including boxes of different categories on the same image.
        self.csv["ActiveTuberculosis"] = self.csv["id"].map(lambda x: float(any(a["category_id"] == cat_map["ActiveTuberculosis"] for a in ann_dict[x])))
        self.csv["ObsoletePulmonaryTuberculosis"] = self.csv["id"].map(lambda x: float(any(a["category_id"] == cat_map["ObsoletePulmonaryTuberculosis"] for a in ann_dict[x])))
        self.csv["PulmonaryTuberculosis"] = self.csv["id"].map(lambda x: float(any(a["category_id"] == cat_map["PulmonaryTuberculosis"] for a in ann_dict[x])))
        # An image is positive for the "Tuberculosis" superclass if it has any annotation
        # belonging to a category under the Tuberculosis supercategory. Lets users who
        # only care about TB vs. non-TB use this column and drop the granular ones.
        tb_cat_ids = {cat["id"] for cat in data["categories"] if cat["supercategory"] == "Tuberculosis"}
        self.csv["Tuberculosis"] = self.csv["id"].map(lambda x: float(any(a["category_id"] in tb_cat_ids for a in ann_dict[x])))

        self.labels = self.csv[self.pathologies].values.astype(np.float32)

    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self),self.data_aug)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        sample["bbox"] = self.csv['bbox'].iloc[idx]
        imgid = self.csv['file_name'].iloc[idx]
        img_path = os.path.join(self.imgpath, "imgs", imgid)
        img = imread(img_path)
        sample["img"] = normalize(img, maxval=255, reshape=True)
        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)
        return sample
    
class SIIM_Pneumothorax_Dataset(Dataset):
    """SIIM-ACR Pneumothorax Segmentation dataset

    The training corpus from the 2019 SIIM-ACR Pneumothorax Segmentation
    Kaggle challenge. It contains 12,954 chest X-ray images in DICOM format
    along with run-length-encoded (RLE) segmentation masks that delineate
    pneumothorax (collapsed lung) regions. Images without pneumothorax carry
    a mask value of ``-1``.

    **Pathologies (1):** Pneumothorax.

    Per-image segmentation masks are available via ``pathology_masks=True``.
    Requires ``pydicom`` to read the ``.dcm`` image files.

    .. note::
        Some training images have multiple annotations from different
        radiologists; all annotations are combined into a single mask.

    Challenge site:
        https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

    Download DICOM images:
        https://academictorrents.com/details/6ef7c6d039e85152c4d0f31d83fa70edc4aba088
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 transform=None,
                 data_aug=None,
                 seed=0,
                 unique_patients=True,
                 pathology_masks=False
                 ):
        super(SIIM_Pneumothorax_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "siim-pneumothorax-train-rle.csv.gz")
        else:
            self.csvpath = csvpath

        self.csv = pd.read_csv(self.csvpath)

        self.pathologies = ["Pneumothorax"]

        labels = [self.csv[" EncodedPixels"] != "-1"]
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.csv = self.csv.reset_index()

        self.csv["has_masks"] = self.csv[" EncodedPixels"] != "-1"

        # To figure out the paths
        # TODO: make faster
        if not ("siim_file_map" in _cache_dict):
            file_map = {}
            for root, directories, files in os.walk(self.imgpath, followlinks=False):
                for filename in files:
                    filePath = os.path.join(root, filename)
                    file_map[filename] = filePath
            _cache_dict["siim_file_map"] = file_map
        self.file_map = _cache_dict["siim_file_map"]

    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self), self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['ImageId'].iloc[idx]
        img_path = self.file_map[imgid + ".dcm"]

        try:
            import pydicom
        except ImportError as e:
            raise Exception("Please install pydicom to work with this dataset")
        img = pydicom.filereader.dcmread(img_path).pixel_array

        sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_pathology_mask_dict(imgid, sample["img"].shape[2])

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_pathology_mask_dict(self, image_name, this_size):

        base_size = 1024
        images_with_masks = self.csv[np.logical_and(self.csv["ImageId"] == image_name,
                                                    self.csv[" EncodedPixels"] != "-1")]
        path_mask = {}

        # From kaggle code
        def rle2mask(rle, width, height):
            mask = np.zeros(width * height)
            array = np.asarray([int(x) for x in rle.split()])
            starts = array[0::2]
            lengths = array[1::2]

            current_position = 0
            for index, start in enumerate(starts):
                current_position += start
                mask[current_position:current_position + lengths[index]] = 1
                current_position += lengths[index]

            return mask.reshape(width, height)

        if len(images_with_masks) > 0:
            # Using a for loop so it is consistent with the other code
            for patho in ["Pneumothorax"]:
                mask = np.zeros([this_size, this_size])

                # don't add masks for labels we don't have
                if patho in self.pathologies:

                    for i in range(len(images_with_masks)):
                        row = images_with_masks.iloc[i]
                        mask = rle2mask(row[" EncodedPixels"], base_size, base_size)
                        mask = mask.T
                        mask = skimage.transform.resize(mask, (this_size, this_size), mode='constant', order=0)
                        mask = mask.round()  # make 0,1

                # reshape so image resizing works
                mask = mask[None, :, :]

                path_mask[self.pathologies.index(patho)] = mask

        return path_mask


class VinBrain_Dataset(Dataset):
    """VinDr-CXR dataset

    A large chest X-ray dataset collected at two major hospitals in Vietnam
    (Hanoi Medical University Hospital and Bach Mai Hospital), annotated by
    17 experienced radiologists. The training set contains 15,000 DICOM
    images with bounding-box labels covering 14 thoracic abnormalities and a
    "No finding" class.

    **Pathologies (14):** Aortic Enlargement, Atelectasis, Calcification,
    Cardiomegaly, Consolidation, Effusion, ILD, Infiltration, Lesion,
    Lung Opacity, Nodule/Mass, Pleural Thickening, Pneumothorax, Pulmonary
    Fibrosis.

    Per-image bounding-box masks are available via ``pathology_masks=True``.
    Requires ``pydicom`` to read the ``.dicom`` image files.

    Example::

        d_vin = xrv.datasets.VinBrain_Dataset(
            imgpath=".../train",
            csvpath=".../train.csv"
        )

    Citation:
        Nguyen HQ, Lam K, Le LT, et al.
        VinDr-CXR: An open dataset of chest X-rays with radiologist's
        annotations.
        *arXiv:2012.15029*, 2020.
        http://arxiv.org/abs/2012.15029

    Challenge site:
        https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 views=None,
                 transform=None,
                 data_aug=None,
                 seed=0,
                 pathology_masks=False
                 ):
        super(VinBrain_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "vinbigdata-train.csv.gz")
        else:
            self.csvpath = csvpath

        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        self.views = views

        self.pathologies = ['Aortic enlargement',
                            'Atelectasis',
                            'Calcification',
                            'Cardiomegaly',
                            'Consolidation',
                            'ILD',
                            'Infiltration',
                            'Lung Opacity',
                            'Nodule/Mass',
                            'Lesion',
                            'Effusion',
                            'Pleural_Thickening',
                            'Pneumothorax',
                            'Pulmonary Fibrosis']

        self.pathologies = sorted(np.unique(self.pathologies))

        self.mapping = dict()
        self.mapping["Pleural_Thickening"] = ["Pleural thickening"]
        self.mapping["Effusion"] = ["Pleural effusion"]

        # Load data
        self.check_paths_exist()
        self.rawcsv = pd.read_csv(self.csvpath)
        self.csv = pd.DataFrame(self.rawcsv.groupby("image_id")["class_name"].apply(lambda x: "|".join(np.unique(x))))

        self.csv["has_masks"] = self.csv.class_name != "No finding"

        labels = []
        for pathology in self.pathologies:
            mask = self.csv["class_name"].str.lower().str.contains(pathology.lower())
            if pathology in self.mapping:
                for syn in self.mapping[pathology]:
                    mask |= self.csv["class_name"].str.lower().str.contains(syn.lower())
            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.csv = self.csv.reset_index()

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['image_id'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid + ".dicom")

        try:
            import pydicom
        except ImportError as e:
            raise Exception("Please install pydicom to work with this dataset")
        from pydicom.pixel_data_handlers.util import apply_modality_lut
        dicom_obj = pydicom.filereader.dcmread(img_path)
        img = apply_modality_lut(dicom_obj.pixel_array, dicom_obj)
        img = pydicom.pixel_data_handlers.apply_windowing(img, dicom_obj)

        # Photometric Interpretation to see if the image needs to be inverted
        mode = dicom_obj[0x28, 0x04].value
        bitdepth = dicom_obj[0x28, 0x101].value

        # hack!
        if img.max() < 256:
            bitdepth = 8

        if mode == "MONOCHROME1":
            img = -1 * img + 2**float(bitdepth)
        elif mode == "MONOCHROME2":
            pass
        else:
            raise Exception("Unknown Photometric Interpretation mode")

        sample["img"] = normalize(img, maxval=2**float(bitdepth), reshape=True)

        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_mask_dict(self, image_name, this_size):

        c, h, w = this_size

        path_mask = {}
        rows = self.rawcsv[self.rawcsv.image_id.str.contains(image_name)]

        for i, pathology in enumerate(self.pathologies):
            for group_name, df_group in rows.groupby("class_name"):
                if (group_name.lower() == pathology.lower()) or ((pathology in self.mapping) and (group_name in self.mapping[pathology])):

                    mask = np.zeros([h, w])
                    for idx, row in df_group.iterrows():
                        mask[int(row.y_min):int(row.y_max), int(row.x_min):int(row.x_max)] = 1

                    path_mask[i] = mask[None, :, :]

        return path_mask


class StonyBrookCOVID_Dataset(Dataset):
    """Stony Brook COVID-19 Radiographic Assessment of Lung Opacity (RALO) dataset

    A dataset of chest X-rays from COVID-19 positive patients collected at
    Stony Brook University Hospital. Each image is scored for the geographic
    extent and opacity severity of lung involvement using the RALO scoring
    system. Labels are continuous scores rather than binary pathology labels.

    **Pathologies (2):** Geographic Extent, Lung Opacity.

    .. note::
        Both ``imgpath`` (path to ``CXR_images_scored/``) and ``csvpath``
        (path to ``ralo-dataset-metadata.csv``) must be provided. All images
        are AP view.

    Citation:
        Goldgof G, et al.
        Stony Brook Medicine COVID-19 Positive Cases.
        *Zenodo*, 2021.
        https://doi.org/10.5281/zenodo.4633999
    """

    def __init__(self,
                 imgpath,  # path to CXR_images_scored
                 csvpath,  # path to ralo-dataset-metadata.csv
                 transform=None,
                 data_aug=None,
                 views=["AP"],
                 seed=0
                 ):
        super(StonyBrookCOVID_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, skiprows=1)
        self.MAXVAL = 255  # Range [0 255]

        self.pathologies = ["Geographic Extent", "Lung Opacity"]

        self.csv["Geographic Extent"] = (self.csv["Total GEOGRAPHIC"] + self.csv["Total GEOGRAPHIC.1"]) / 2
        self.csv["Lung Opacity"] = (self.csv["Total OPACITY"] + self.csv["Total OPACITY.1"]) / 2

        labels = []
        labels.append(self.csv["Geographic Extent"])
        labels.append(self.csv["Lung Opacity"])

        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # add consistent csv values

        # offset_day_int

        date_col = self.csv["Exam_DateTime"].str.split("_", expand=True)[0]
        dt = pd.to_datetime(date_col, format="%Y%m%d")
        self.csv["offset_day_int"] = dt.astype(np.int64) // 10**9 // 86400

        # patientid
        self.csv["patientid"] = self.csv["Subject_ID"].astype(str)

        # all the images are AP according to the article.
        self.csv["view"] = "AP"
        self.limit_to_selected_views(views)

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        img_path = os.path.join(self.imgpath, str(idx) + ".jpg")
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class ObjectCXR_Dataset(Dataset):
    """Object-CXR foreign object detection dataset

    A challenge dataset from MIDL 2020 containing 10,000 frontal chest X-ray
    images: 5,000 with at least one foreign object present and 5,000 without.
    Images were collected from township hospitals in China via a telemedicine
    platform. Foreign objects are annotated with bounding boxes, ellipses, or
    pixel masks depending on object shape.

    **Pathologies (1):** Foreign Object.

    .. note::
        Images are stored inside a ZIP archive. Pass the path to the ZIP file
        as ``imgzippath`` and the annotation CSV path as ``csvpath``.

    Challenge website:
        https://jfhealthcare.github.io/object-CXR/

    Download images and annotations:
        https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5
        https://archive.org/download/object-CXR/object-CXR/
    """

    def __init__(self,
                 imgzippath,
                 csvpath,
                 transform=None,
                 data_aug=None,
                 seed=0
                 ):
        super(ObjectCXR_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgzippath = imgzippath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug
        self.views = []
        self.pathologies = ['Foreign Object']

        # Load data
        self.csv = pd.read_csv(self.csvpath)

        labels = []
        labels.append(~self.csv["annotation"].isnull())
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.csv = self.csv.reset_index()

        self.csv["has_masks"] = ~self.csv["annotation"].isnull()

        self.imgzip = zipfile.ZipFile(self.imgzippath)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        imgid = self.csv.iloc[idx]["image_name"]

        with zipfile.ZipFile(self.imgzippath).open("train/" + imgid) as file:
            sample["img"] = imageio.imread(file.read())

        sample["img"] = normalize(sample["img"], maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class ToPILImage(object):
    def __init__(self):
        self.to_pil = transforms.ToPILImage(mode="F")

    def __call__(self, x):
        return self.to_pil(x[0])


class XRayResizer(object):
    """Resize an image to a specific size"""

    def __init__(self, size: int, engine="skimage"):
        self.size = size
        self.engine = engine

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), mode='constant', preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            import cv2  # pip install opencv-python
            return cv2.resize(img[0, :, :],
                              (self.size, self.size),
                              interpolation=cv2.INTER_AREA
                              ).reshape(1, self.size, self.size).astype(np.float32)
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")


class XRayCenterCrop(object):
    """Perform a center crop on the long dimension of the input image"""

    def crop_center(self, img: np.ndarray) -> np.ndarray:
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.crop_center(img)


class CovariateDataset(Dataset):
    """A covariate shift between two data distributions arises when some
    extraneous variable confounds with the variables of interest in the first
    dataset differently than in the second [Moreno-Torres et al., 2012].
    Covariate shifts between the training and test distribution in a machine
    learning setting can lead to models which generalize poorly, and this
    phenomenon is commonly observed in CXR models trained on a small dataset
    and deployed on another one [Zhao et al., 2019; DeGrave et al., 2020]. We
    provide tools to simulate covariate shifts in these datasets so
    researchers can evaluate the susceptibility of their models to these
    shifts, or explore mitigation strategies.

    .. code-block:: python

        d = xrv.datasets.CovariateDataset(
            d1 = # dataset1 with a specific condition.
            d1_target = # target label to predict.
            d2 = # dataset2 with a specific condition.
            d2_target = #target label to predict.
            mode="train", # train, valid, or test.
            ratio=0.75
        )

    .. image:: _static/CovariateDataset-Diagram.png

    The class xrv.datasets.CovariateDataset takes two datasets and two arrays
    representing the labels. It returns samples for the output classes with a
    specified ratio of examples from each dataset, thereby introducing a
    correlation between any dataset-specific nuisance features and the output
    label. This simulates a covariate shift. The test split can be set up
    with a different ratio than the training split; this setup has been shown
    to both decrease generalization performance and exacerbate incorrect
    feature attribution [Viviano et al., 2020]. See Figure 4 for a
    visualization of the effect the ratio parameter has on the mean class
    difference when correlating the view (each dataset) with the target
    label. The effect seen with low ratios is due to the majority of the
    positive labels being drawn from the first dataset, where in the high
    ratios, the majority of the positive labels are drawn from the second
    dataset. With any ratio, the number of samples returned will be the same
    in order to provide controlled experiments. The dataset has 3 modes,
    train sampled using the provided ratio and the valid and test dataset are
    sampled using 1−ratio.

    An example of the mean class difference drawn from the COVID-19 dataset
    at different covariate ratios. Here, the first COVID-19 dataset consisted
    of only AP images, whereas the second dataset consisted of only PA
    images. The third row shows, for each ratio, the difference in the class
    means, demonstrating the effect of sampling images from the two views on
    the perceived class difference. The fourth row shows the difference
    between each ratio’s difference image, and the difference image with a
    ratio of 0.5 (balanced sampling from all views).

    .. image:: _static/covariate.png

    Citation:

    Viviano, J. D., Simpson, B., Dutil, F., Bengio, Y., & Cohen, J. P. (2020).
    Saliency is a Possible Red Herring When Diagnosing Poor Generalization.
    International Conference on Learning Representations (ICLR).
    https://arxiv.org/abs/1910.00199
    """

    def __init__(self,
                 d1, d1_target,
                 d2, d2_target,
                 ratio=0.5,
                 mode="train",
                 seed=0,
                 nsamples=None,
                 splits=[0.5, 0.25, 0.25],
                 verbose=False
                 ):
        super(CovariateDataset, self).__init__()

        self.splits = np.array(splits)
        self.d1 = d1
        self.d1_target = d1_target
        self.d2 = d2
        self.d2_target = d2_target

        assert mode in ['train', 'valid', 'test']
        assert np.sum(self.splits) == 1.0

        np.random.seed(seed)  # Reset the seed so all runs are the same.

        all_imageids = np.concatenate([np.arange(len(self.d1)),
                                       np.arange(len(self.d2))]).astype(int)

        all_idx = np.arange(len(all_imageids)).astype(int)

        all_labels = np.concatenate([d1_target,
                                     d2_target]).astype(int)

        all_site = np.concatenate([np.zeros(len(self.d1)),
                                   np.ones(len(self.d2))]).astype(int)

        idx_sick = all_labels == 1
        n_per_category = np.min([sum(idx_sick[all_site == 0]),
                                 sum(idx_sick[all_site == 1]),
                                 sum(~idx_sick[all_site == 0]),
                                 sum(~idx_sick[all_site == 1])])

        all_csv = pd.concat([d1.csv, d2.csv])
        all_csv['site'] = all_site
        all_csv['label'] = all_labels

        if verbose:
            print("n_per_category={}".format(n_per_category))

        all_0_neg = all_idx[np.where((all_site == 0) & (all_labels == 0))]
        all_0_neg = np.random.choice(all_0_neg, n_per_category, replace=False)
        all_0_pos = all_idx[np.where((all_site == 0) & (all_labels == 1))]
        all_0_pos = np.random.choice(all_0_pos, n_per_category, replace=False)
        all_1_neg = all_idx[np.where((all_site == 1) & (all_labels == 0))]
        all_1_neg = np.random.choice(all_1_neg, n_per_category, replace=False)
        all_1_pos = all_idx[np.where((all_site == 1) & (all_labels == 1))]
        all_1_pos = np.random.choice(all_1_pos, n_per_category, replace=False)

        # TRAIN
        train_0_neg = np.random.choice(
            all_0_neg, int(n_per_category * ratio * splits[0] * 2), replace=False)
        train_0_pos = np.random.choice(
            all_0_pos, int(n_per_category * (1 - ratio) * splits[0] * 2), replace=False)
        train_1_neg = np.random.choice(
            all_1_neg, int(n_per_category * (1 - ratio) * splits[0] * 2), replace=False)
        train_1_pos = np.random.choice(
            all_1_pos, int(n_per_category * ratio * splits[0] * 2), replace=False)

        # REDUCE POST-TRAIN
        all_0_neg = np.setdiff1d(all_0_neg, train_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, train_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, train_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, train_1_pos)

        if verbose:
            print("TRAIN (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                ratio,
                len(train_0_neg) + len(train_1_neg),
                len(train_0_pos) + len(train_1_pos),
                len(train_0_pos),
                len(train_0_neg),
                len(train_1_pos),
                len(train_1_neg)))

        # VALID
        valid_0_neg = np.random.choice(
            all_0_neg, int(n_per_category * (1 - ratio) * splits[1] * 2), replace=False)
        valid_0_pos = np.random.choice(
            all_0_pos, int(n_per_category * ratio * splits[1] * 2), replace=False)
        valid_1_neg = np.random.choice(
            all_1_neg, int(n_per_category * ratio * splits[1] * 2), replace=False)
        valid_1_pos = np.random.choice(
            all_1_pos, int(n_per_category * (1 - ratio) * splits[1] * 2), replace=False)

        # REDUCE POST-VALID
        all_0_neg = np.setdiff1d(all_0_neg, valid_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, valid_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, valid_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, valid_1_pos)

        if verbose:
            print("VALID (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                1 - ratio,
                len(valid_0_neg) + len(valid_1_neg),
                len(valid_0_pos) + len(valid_1_pos),
                len(valid_0_pos),
                len(valid_0_neg),
                len(valid_1_pos),
                len(valid_1_neg)))

        # TEST
        test_0_neg = all_0_neg
        test_0_pos = all_0_pos
        test_1_neg = all_1_neg
        test_1_pos = all_1_pos

        if verbose:
            print("TEST (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                1 - ratio,
                len(test_0_neg) + len(test_1_neg),
                len(test_0_pos) + len(test_1_pos),
                len(test_0_pos),
                len(test_0_neg),
                len(test_1_pos),
                len(test_1_neg)))

        def _reduce_nsamples(nsamples, a, b, c, d):
            if nsamples:
                a = a[:int(np.floor(nsamples / 4))]
                b = b[:int(np.ceil(nsamples / 4))]
                c = c[:int(np.ceil(nsamples / 4))]
                d = d[:int(np.floor(nsamples / 4))]

            return (a, b, c, d)

        if mode == "train":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, train_0_neg, train_0_pos, train_1_neg, train_1_pos)
        elif mode == "valid":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, valid_0_neg, valid_0_pos, valid_1_neg, valid_1_pos)
        elif mode == "test":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, test_0_neg, test_0_pos, test_1_neg, test_1_pos)
        else:
            raise Exception("unknown mode")

        self.select_idx = np.concatenate([a, b, c, d])
        self.imageids = all_imageids[self.select_idx]
        self.pathologies = ["Custom"]
        self.labels = all_labels[self.select_idx].reshape(-1, 1)
        self.site = all_site[self.select_idx]
        self.csv = all_csv.iloc[self.select_idx]

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.imageids)

    def __getitem__(self, idx):

        if self.site[idx] == 0:
            dataset = self.d1
        else:
            dataset = self.d2

        sample = dataset[self.imageids[idx]]

        # Replace the labels with the specific label we focus on
        sample["lab-old"] = sample["lab"]
        sample["lab"] = self.labels[idx]

        sample["site"] = self.site[idx]

        return sample
