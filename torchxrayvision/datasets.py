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
import skimage
from typing import Dict, List
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
        self.csv.view.fillna("UNKNOWN", inplace=True)

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

    ChestX-ray dataset comprises 112,120 frontal-view X-ray images of 30,
    805 unique patients with the text-mined fourteen disease image labels (
    where each image can have multi-labels), mined from the associated
    radiological reports using natural language processing. Fourteen common
    thoracic pathologies include Atelectasis, Consolidation, Infiltration,
    Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia,
    Pleural_thickening, Cardiomegaly, Nodule, Mass and Hernia, which is an
    extension of the 8 common disease patterns listed in our CVPR2017 paper.
    Note that original radiology reports (associated with these chest x-ray
    studies) are not meant to be publicly shared for many reasons. The
    text-mined disease labels are expected to have accuracy >90%.Please find
    more details and benchmark performance of trained models based on 14
    disease labels in our arxiv paper: https://arxiv.org/abs/1705.02315

    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

    Download resized (224x224) images here:
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
    """RSNA Pneumonia Detection Challenge

    Citation:

    Augmenting the National Institutes of Health Chest Radiograph Dataset
    with Expert Annotations of Possible Pneumonia. Shih, George, Wu,
    Carol C., Halabi, Safwan S., Kohli, Marc D., Prevedello, Luciano M.,
    Cook, Tessa S., Sharma, Arjun, Amorosa, Judith K., Arteaga, Veronica,
    Galperin-Aizenberg, Maya, Gill, Ritu R., Godoy, Myrna C.B., Hobbs,
    Stephen, Jeudy, Jean, Laroia, Archana, Shah, Palmi N., Vummidi, Dharshan,
    Yaddanapudi, Kavitha, and Stein, Anouk. Radiology: Artificial
    Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.

    More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018

    Challenge site:
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

    JPG files stored here:
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
    """A relabelling of a subset of images from the NIH dataset.  The data
    tables should be applied against an NIH download.  A test and validation
    split are provided in the original.  They are combined here, but one or
    the other can be used by providing the original csv to the csvpath
    argument.

    Citation:

    Chest Radiograph Interpretation with Deep Learning Models: Assessment
    with Radiologist-adjudicated Reference Standards and Population-adjusted
    Evaluation Anna Majkowska, Sid Mittal, David F. Steiner, Joshua J.
    Reicher, Scott Mayer McKinney, Gavin E. Duggan, Krish Eswaran, Po-Hsuan
    Cameron Chen, Yun Liu, Sreenivasa Raju Kalidindi, Alexander Ding, Greg S.
    Corrado, Daniel Tse, and Shravya Shetty. Radiology 2020

    https://pubs.rsna.org/doi/10.1148/radiol.2019191293

    NIH data can be downloaded here:
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
    """PadChest dataset from the Hospital San Juan de Alicante - University of
    Alicante

    Note that images with null labels (as opposed to normal), and images that
    cannot be properly loaded (listed as 'missing' in the code) are excluded,
    which makes the total number of available images slightly less than the
    total number of image files.

    Citation:

    PadChest: A large chest x-ray image dataset with multi-label annotated
    reports. Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria
    de la Iglesia-Vayá. arXiv preprint, 2019. https://arxiv.org/abs/1901.07441

    Dataset website:
    http://bimcv.cipf.es/bimcv-projects/padchest/

    Download full size images here:
    https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850

    Download resized (224x224) images here (recropped):
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
                   "216840111366964012989926673512011083134050913_00-168-009.png", # broken PNG file (chunk b'\x00\x00\x00\x00')
                   "216840111366964012373310883942009152114636712_00-102-045.png", # "OSError: image file is truncated"
                   "216840111366964012819207061112010281134410801_00-129-131.png", # "OSError: image file is truncated"
                   "216840111366964012487858717522009280135853083_00-075-001.png", # "OSError: image file is truncated"
                   "216840111366964012989926673512011151082430686_00-157-045.png", # broken PNG file (chunk b'\x00\x00\x00\x00')
                   "216840111366964013686042548532013208193054515_02-026-007.png", # "OSError: image file is truncated"
                   "216840111366964013590140476722013058110301622_02-056-111.png", # "OSError: image file is truncated"
                   "216840111366964013590140476722013043111952381_02-065-198.png", # "OSError: image file is truncated"
                   "216840111366964013829543166512013353113303615_02-092-190.png", # "OSError: image file is truncated"
                   "216840111366964013962490064942014134093945580_01-178-104.png", # "OSError: image file is truncated"
                  ]
        self.csv = self.csv[~self.csv["ImageID"].isin(missing)]

        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Filter out age < 10 (paper published 2019)
        self.csv = self.csv[(2019 - self.csv.PatientBirth > 10)]

        # Get our classes.
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    #print("mapping", syn)
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
    """CheXpert Dataset

    Citation:

    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and
    Expert Comparison. Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko,
    Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad
    Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong,
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson,
    Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng.
    https://arxiv.org/abs/1901.07031

    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/

    A small validation set is provided with the data as well, but is so tiny,
    it is not included here.
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
        self.csv['Age'][(self.csv['Age'] == 0)] = None

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
        imgid = imgid.replace("CheXpert-v1.0-small/", "")
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class MIMIC_Dataset(Dataset):
    """MIMIC-CXR Dataset

    Citation:

    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY,
    Mark RG, Horng S. MIMIC-CXR: A large publicly available database of
    labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.

    https://arxiv.org/abs/1901.07042

    Dataset website here:
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
    """OpenI Dataset

    Dina Demner-Fushman, Marc D. Kohli, Marc B. Rosenman, Sonya E. Shooshan,
    Laritza Rodriguez, Sameer Antani, George R. Thoma, and Clement J.
    McDonald. Preparing a collection of radiology examinations for
    distribution and retrieval. Journal of the American Medical Informatics
    Association, 2016. doi: 10.1093/jamia/ocv080.

    Views have been determined by projection using T-SNE.  To use the T-SNE
    view rather than the view defined by the record,
    set use_tsne_derived_view to true.

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
                    #print("mapping", syn)
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

    This dataset currently contains hundreds of frontal view X-rays and is
    the largest public resource for COVID-19 image and prognostic data,
    making it a necessary resource to develop and evaluate tools to aid in
    the treatment of COVID-19. It was manually aggregated from publication
    figures as well as various web based repositories into a machine learning
    (ML) friendly format with accompanying dataloader code. We collected
    frontal and lateral view imagery and metadata such as the time since
    first symptoms, intensive care unit (ICU) status, survival status,
    intubation status, or hospital location. We present multiple possible use
    cases for the data such as predicting the need for the ICU, predicting
    patient survival, and understanding a patient's trajectory during
    treatment.

    Citations:

    COVID-19 Image Data Collection: Prospective Predictions Are the Future
    Joseph Paul Cohen and Paul Morrison and Lan Dao and Karsten Roth and Tim
    Q Duong and Marzyeh Ghassemi arXiv:2006.11988, 2020

    COVID-19 image data collection,
    Joseph Paul Cohen and Paul Morrison and Lan Dao
    arXiv:2003.11597, 2020

    Dataset: https://github.com/ieee8023/covid-chestxray-dataset

    Paper: https://arxiv.org/abs/2003.11597
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
        """
        Args:
            imgpath: Path to the directory containing images
            csvpath: Path to the image directory
        """

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

                mask = (mask == 255).astype(np.float)
                # Reshape so image resizing works
                mask = mask[None, :, :]

                semantic_masks["Lungs"] = mask

        return semantic_masks


class NLMTB_Dataset(Dataset):
    """National Library of Medicine Tuberculosis Datasets

    https://lhncbc.nlm.nih.gov/publication/pub9931
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

    Note that each dataset should be loaded separately by this class (they
    may be merged afterwards).  All images are of view PA.

    Jaeger S, Candemir S, Antani S, Wang YX, Lu PX, Thoma G. Two public chest
    X-ray datasets for computer-aided screening of pulmonary diseases. Quant
    Imaging Med Surg. 2014 Dec;4(6):475-7. doi:
    10.3978/j.issn.2223-4292.2014.11.20. PMID: 25525580; PMCID: PMC4256233.

    Download Links:
    Montgomery County
    https://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33
    http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip

    Shenzhen
    https://academictorrents.com/details/462728e890bd37c05e9439c885df7afc36209cc8
    http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
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


class SIIM_Pneumothorax_Dataset(Dataset):
    """SIIM Pneumothorax Dataset

    https://academictorrents.com/details/6ef7c6d039e85152c4d0f31d83fa70edc4aba088
    https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

    "The data is comprised of images in DICOM format and annotations in the
    form of image IDs and run-length-encoded (RLE) masks. Some of the images
    contain instances of pneumothorax (collapsed lung), which are indicated
    by encoded binary masks in the annotations. Some training images have
    multiple annotations. Images without pneumothorax have a mask value of -1."
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
    """VinBrain Dataset

    .. code-block:: python

        d_vin = xrv.datasets.VinBrain_Dataset(
            imgpath=".../train",
            csvpath=".../train.csv"
        )

    Nguyen, H. Q., Lam, K., Le, L. T., Pham, H. H., Tran, D. Q., Nguyen,
    D. B., Le, D. D., Pham, C. M., Tong, H. T. T., Dinh, D. H., Do, C. D.,
    Doan, L. T., Nguyen, C. N., Nguyen, B. T., Nguyen, Q. V., Hoang, A. D.,
    Phan, H. N., Nguyen, A. T., Ho, P. H., … Vu, V. (2020). VinDr-CXR: An
    open dataset of chest X-rays with radiologist’s annotations.
    http://arxiv.org/abs/2012.15029

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
                if (group_name == pathology) or ((pathology in self.mapping) and (group_name in self.mapping[pathology])):

                    mask = np.zeros([h, w])
                    for idx, row in df_group.iterrows():
                        mask[int(row.y_min):int(row.y_max), int(row.x_min):int(row.x_max)] = 1

                    path_mask[i] = mask[None, :, :]

        return path_mask


class StonyBrookCOVID_Dataset(Dataset):
    """Stonybrook Radiographic Assessment of Lung Opacity Score Dataset

    https://doi.org/10.5281/zenodo.4633999

    Citation will be set soon.
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
    """ObjectCXR Dataset

    "We provide a large dataset of chest X-rays with strong annotations of
    foreign objects, and the competition for automatic detection of foreign
    objects. Specifically, 5000 frontal chest X-ray images with foreign
    objects presented and 5000 frontal chest X-ray images without foreign
    objects are provided. All the chest X-ray images were filmed in township
    hospitals in China and collected through our telemedicine platform.
    Foreign objects within the lung field of each chest X-ray are annotated
    with bounding boxes, ellipses or masks depending on the shape of the
    objects."

    Challenge dataset from MIDL2020

    https://jfhealthcare.github.io/object-CXR/

    https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5
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
        if 'cv2' in sys.modules:
            print("Setting XRayResizer engine to cv2 could increase performance.")

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
