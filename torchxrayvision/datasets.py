from PIL import Image
from os.path import join
from skimage.io import imread, imsave
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os,sys,os.path
import pandas as pd
import pickle
import pydicom
import skimage
import glob
import collections
import pprint
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
import skimage.transform
import warnings
import tarfile
import zipfile
import random

default_pathologies = [  'Atelectasis',
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

thispath = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(thispath,"data")

# this is for caching small things for speed
_cache_dict = {}

def normalize(sample, maxval):
    """Scales images to be roughly [-1024 1024]."""
    
    if sample.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(sample.max(), maxval))
    
    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    #sample = sample / np.std(sample)
    return sample

def relabel_dataset(pathologies, dataset, silent=False):
    """
    Reorder, remove, or add (nans) to a dataset's labels.
    Use this to align with the output of a network.
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
            new_labels.append(dataset.labels[:,pathology_idx])
        else:
            if not silent:
                print("{} doesn't exist. Adding nans instead.".format(pathology))
            values = np.empty(dataset.labels.shape[0])
            values.fill(np.nan)
            new_labels.append(values)
    new_labels = np.asarray(new_labels).T
    
    dataset.labels = new_labels
    dataset.pathologies = pathologies

class Dataset():
    def __init__(self):
        pass
    def totals(self):
        counts = [dict(collections.Counter(items[~np.isnan(items)]).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies,counts))
    def __repr__(self):
        pprint.pprint(self.totals())
        return self.string()
    def check_paths_exist(self):
        #if self.imagezipfile is not None:
            
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")
        
    
class Merge_Dataset(Dataset):
    def __init__(self, datasets, seed=0, label_concat=False):
        super(Merge_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.datasets = datasets
        self.length = 0
        self.pathologies = datasets[0].pathologies
        self.which_dataset = np.zeros(0)
        self.offset = np.zeros(0)
        currentoffset = 0
        for i, dataset in enumerate(datasets):
            self.which_dataset = np.concatenate([self.which_dataset, np.zeros(len(dataset))+i])
            self.length += len(dataset)
            self.offset = np.concatenate([self.offset, np.zeros(len(dataset))+currentoffset])
            currentoffset += len(dataset)
            if dataset.pathologies != self.pathologies:
                raise Exception("incorrect pathology alignment")
                
        if hasattr(datasets[0], 'labels'):
            self.labels = np.concatenate([d.labels for d in datasets])
        else:
            print("WARN: not adding .labels")
        
        self.which_dataset = self.which_dataset.astype(int)
        
        if label_concat:
            new_labels = np.zeros([self.labels.shape[0], self.labels.shape[1]*len(datasets)])*np.nan
            for i, shift in enumerate(self.which_dataset):
                size = self.labels.shape[1]
                new_labels[i,shift*size:shift*size+size] = self.labels[i]
            self.labels = new_labels
            
        try:
            self.csv = pd.concat([d.csv for d in datasets])
        except:
            print("Could not merge dataframes (.csv not available):", sys.exc_info()[0])
        
        self.csv = self.csv.reset_index()

    def string(self):
        s = self.__class__.__name__ + " num_samples={}\n".format(len(self))
        for i, d in enumerate(self.datasets):
            if i < len(self.datasets)-1:
                s += "├{} ".format(i) + d.string().replace("\n","\n|  ") + "\n"
            else:
                s += "└{} ".format(i) + d.string().replace("\n","\n   ") + "\n"
        return s
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = self.datasets[int(self.which_dataset[idx])][idx  - int(self.offset[idx])]
        item["lab"] = self.labels[idx]
        item["source"] = self.which_dataset[idx]
        return item
        
class FilterDataset(Dataset):
    def __init__(self, dataset, labels=None):
        super(FilterDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies
        
#         self.idxs = np.where(np.nansum(dataset.labels, axis=1) > 0)[0]
        
        self.idxs = []
        if labels:
            for label in labels:
                print("filtering for ", label)
                
                self.idxs += list(np.where(dataset.labels[:,list(dataset.pathologies).index(label)] == 1)[0])
#             singlelabel = np.nanargmax(dataset.labels[self.idxs], axis=1)
#             subset = [k in labels for k in singlelabel]
#             self.idxs = self.idxs[np.array(subset)]
        
        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]
        
    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n","\n  ")
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

class SubsetDataset(Dataset):
    def __init__(self, dataset, idxs=None):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies
        
        self.idxs = idxs
        
        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]
        
        self.csv = self.csv.reset_index(drop=True)
        
        if hasattr(self.dataset, 'which_dataset'):
            self.which_dataset = self.dataset.which_dataset[self.idxs]
    
    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n","\n  ")
    
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]

    
class NIH_Dataset(Dataset):
    """
    NIH ChestX-ray8 dataset

    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
    
    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a
    
    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """
    def __init__(self, imgpath, 
                 csvpath=os.path.join(datapath, "Data_Entry_2017_v2020.csv.gz"),
                 bbox_list_path=os.path.join(datapath, "BBox_List_2017.csv.gz"),
                 views=["PA"],
                 transform=None, 
                 data_aug=None, 
                 nrows=None, 
                 seed=0,
                 pure_labels=False, 
                 unique_patients=True,
                 normalize=True,
                 pathology_masks=False):
        
        super(NIH_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        
        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia"]
        
        self.pathologies = sorted(self.pathologies)
        
        self.normalize = normalize
        # Load data
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]
        
        if type(views) is not list:
            views = [views]
        self.views = views
        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.csv = self.csv[self.csv["view"].isin(self.views)]
        
        # Remove multi-finding images.
        if pure_labels:
            self.csv = self.csv[~self.csv["Finding Labels"].str.contains("\|")]
        
        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()
        
        self.csv = self.csv.reset_index()
        
        ####### pathology masks ########
        # load nih pathology masks
        self.pathology_maskscsv = pd.read_csv(bbox_list_path, 
                names=["Image Index","Finding Label","x","y","w","h","_1","_2","_3"],
               skiprows=1)
        
        # change label name to match
        self.pathology_maskscsv["Finding Label"][self.pathology_maskscsv["Finding Label"] == "Infiltrate"] = "Infiltration"
        self.csv["has_masks"] = self.csv["Image Index"].isin(self.pathology_maskscsv["Image Index"])
        ####### pathology masks ########    
            
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)
            
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        ########## add consistent csv values
        
        # offset_day_int
        #self.csv["offset_day_int"] = 
        
        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)
        

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
        #print(img_path)
        img = imread(img_path)
        if self.normalize:
            img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        sample["img"] = img[None, :, :]                    

        transform_seed = np.random.randint(2147483647)
        
        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])
            
        if self.transform is not None:
            random.seed(transform_seed)
            sample["img"] = self.transform(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.transform(sample["pathology_masks"][i])
  
        if self.data_aug is not None:
            random.seed(transform_seed)
            sample["img"] = self.data_aug(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.data_aug(sample["pathology_masks"][i])
            
        return sample
    
    def get_mask_dict(self, image_name, this_size):

        base_size = 1024
        scale = this_size/base_size

        images_with_masks = self.pathology_maskscsv[self.pathology_maskscsv["Image Index"] == image_name]
        path_mask = {}

        for i in range(len(images_with_masks)):
            row = images_with_masks.iloc[i]
            
            # don't add masks for labels we don't have
            if row["Finding Label"] in self.pathologies:
                mask = np.zeros([this_size,this_size])
                xywh = np.asarray([row.x,row.y,row.w,row.h])
                xywh = xywh*scale
                xywh = xywh.astype(int)
                mask[xywh[1]:xywh[1]+xywh[3],xywh[0]:xywh[0]+xywh[2]] = 1
                
                # resize so image resizing works
                mask = mask[None, :, :] 
                
                path_mask[self.pathologies.index(row["Finding Label"])] = mask
        return path_mask
    
class RSNA_Pneumonia_Dataset(Dataset):
    """
    RSNA Pneumonia Detection Challenge
    
    Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert 
    Annotations of Possible Pneumonia.
    Shih, George, Wu, Carol C., Halabi, Safwan S., Kohli, Marc D., Prevedello, Luciano M., 
    Cook, Tessa S., Sharma, Arjun, Amorosa, Judith K., Arteaga, Veronica, Galperin-Aizenberg, 
    Maya, Gill, Ritu R., Godoy, Myrna C.B., Hobbs, Stephen, Jeudy, Jean, Laroia, Archana, 
    Shah, Palmi N., Vummidi, Dharshan, Yaddanapudi, Kavitha, and Stein, Anouk.  
    Radiology: Artificial Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.
    
    More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
    
    Challenge site:
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
    
    JPG files stored here:
    https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
    """
    def __init__(self, 
                 imgpath, 
                 csvpath=os.path.join(datapath, "kaggle_stage_2_train_labels.csv.zip"),
                 dicomcsvpath=os.path.join(datapath, "kaggle_stage_2_train_images_dicom_headers.csv.gz"),
                 views=["PA"],
                 transform=None, 
                 data_aug=None, 
                 nrows=None, 
                 seed=0,
                 pure_labels=False, 
                 unique_patients=True,
                 normalize=True,
                 pathology_masks=False,
                 extension=".jpg"):

        super(RSNA_Pneumonia_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        
        self.pathologies = ["Pneumonia", "Lung Opacity"]
        
        self.pathologies = sorted(self.pathologies)
        
        self.normalize=normalize
        
        self.extension = extension
        self.use_pydicom=( extension == ".dcm" ) 
        

        # Load data
        self.csvpath = csvpath
        self.raw_csv = pd.read_csv(self.csvpath, nrows=nrows)
        
        # the labels have multiple instances for each mask 
        # so we just need one to get the target label
        self.csv = self.raw_csv.groupby("patientId").first()
        
        self.dicomcsvpath = dicomcsvpath
        self.dicomcsv = pd.read_csv(self.dicomcsvpath, nrows=nrows, index_col="PatientID")
        
        self.csv = self.csv.join(self.dicomcsv, on="patientId")
        
        self.MAXVAL = 255  # Range [0 255]
        
        if type(views) is not list:
            views = [views]
        self.views = views
        # Remove images with view position other than specified
        self.csv["view"] = self.csv['ViewPosition']
        self.csv = self.csv[self.csv["view"].isin(self.views)]
        
        self.csv = self.csv.reset_index()

        # Get our classes.
        self.labels = []
        self.labels.append(self.csv["Target"].values)
        self.labels.append(self.csv["Target"].values) #same labels for both
        
        # set if we have masks
        self.csv["has_masks"] = ~np.isnan(self.csv["x"])
        
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        ########## add consistent csv values
        
        # offset_day_int
        #TODO: merge with NIH metadata to get dates for images
        
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
        #print(img_path)
        if self.use_pydicom:
            img=pydicom.filereader.dcmread(img_path).pixel_array
        else:
            img = imread(img_path)
        if self.normalize:
            img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        sample["img"] = img[None, :, :]                
                               
        transform_seed = np.random.randint(2147483647)
        
        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])
            
        if self.transform is not None:
            random.seed(transform_seed)
            sample["img"] = self.transform(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.transform(sample["pathology_masks"][i])
  
        if self.data_aug is not None:
            random.seed(transform_seed)
            sample["img"] = self.data_aug(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.data_aug(sample["pathology_masks"][i])
            
        return sample
    
    def get_mask_dict(self, image_name, this_size):

        base_size = 1024
        scale = this_size/base_size

        images_with_masks = self.raw_csv[self.raw_csv["patientId"] == image_name]
        path_mask = {}

        # all masks are for both pathologies
        for patho in ["Pneumonia", "Lung Opacity"]:
            
            
            mask = np.zeros([this_size,this_size])
            
            # don't add masks for labels we don't have
            if patho in self.pathologies:
                
                for i in range(len(images_with_masks)):
                    row = images_with_masks.iloc[i]
                    xywh = np.asarray([row.x,row.y,row.width,row.height])
                    xywh = xywh*scale
                    xywh = xywh.astype(int)
                    mask[xywh[1]:xywh[1]+xywh[3],xywh[0]:xywh[0]+xywh[2]] = 1
                
            # resize so image resizing works
            mask = mask[None, :, :] 
                
            path_mask[self.pathologies.index(patho)] = mask
        return path_mask

class NIH_Google_Dataset(Dataset):

    """
    Chest Radiograph Interpretation with Deep Learning Models: Assessment with 
    Radiologist-adjudicated Reference Standards and Population-adjusted Evaluation
    Anna Majkowska, Sid Mittal, David F. Steiner, Joshua J. Reicher, Scott Mayer 
    McKinney, Gavin E. Duggan, Krish Eswaran, Po-Hsuan Cameron Chen, Yun Liu, 
    Sreenivasa Raju Kalidindi, Alexander Ding, Greg S. Corrado, Daniel Tse, and 
    Shravya Shetty. Radiology 2020
    
    https://pubs.rsna.org/doi/10.1148/radiol.2019191293
    """
    
    def __init__(self, imgpath, 
                 csvpath=os.path.join(datapath, "google2019_nih-chest-xray-labels.csv.gz"), 
                 views=["PA"],
                 transform=None, 
                 data_aug=None, 
                 nrows=None, 
                 seed=0,
                 pure_labels=False, 
                 unique_patients=True,
                 normalize=True):

        super(NIH_Google_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        
        self.pathologies = ["Fracture", "Pneumothorax", "Airspace opacity",
                            "Nodule or mass"]
        
        self.pathologies = sorted(self.pathologies)
        
        self.normalize = normalize

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]
        
        if type(views) is not list:
            views = [views]
        self.views = views
        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.csv = self.csv[self.csv["view"].isin(self.views)]

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first().reset_index()
            
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            #if pathology in self.csv.columns:
                #self.csv.loc[pathology] = 0
            mask = self.csv[pathology] == "YES"
                
            self.labels.append(mask.values)
            
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        # rename pathologies
        self.pathologies = np.char.replace(self.pathologies, "Airspace opacity", "Lung Opacity")

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        #print(img_path)
        img = imread(img_path)
        if self.normalize:
            img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]                    
                               
        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)
            
        return {"img":img, "lab":self.labels[idx], "idx":idx}
    
    
class PC_Dataset(Dataset):
    """
    PadChest dataset
    Hospital San Juan de Alicante - University of Alicante
    
    PadChest: A large chest x-ray image dataset with multi-label annotated reports.
    Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria de la Iglesia-Vayá. 
    arXiv preprint, 2019. https://arxiv.org/abs/1901.07441
    
    Dataset website:
    http://bimcv.cipf.es/bimcv-projects/padchest/
    
    Download full size images here:
    https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850
    
    Download resized (224x224) images here (recropped):
    https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797
    """
    def __init__(self, imgpath, 
                 csvpath=os.path.join(datapath, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"), 
                 views=["PA"],
                 transform=None, 
                 data_aug=None,
                 flat_dir=True, 
                 seed=0, 
                 unique_patients=True):

        super(PC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia","Fracture", 
                            "Granuloma", "Flattened Diaphragm", "Bronchiectasis",
                            "Aortic Elongation", "Scoliosis", 
                            "Hilar Enlargement", "Tuberculosis",
                            "Air Trapping", "Costophrenic Angle Blunting", "Aortic Atheromatosis",
                            "Hemidiaphragm Elevation", 
                            "Support Devices", "Tube'"] # the Tube' is intentional
        
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
        mapping["Tube'"] = ["stent'"] ## the ' is to select findings which end in that word
        
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.flat_dir = flat_dir
        self.csvpath = csvpath
        
        self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, low_memory=False)
        self.MAXVAL = 65535

        # standardize view names
        self.csv.loc[self.csv["Projection"].isin(["AP_horizontal"]),"Projection"] = "AP Supine"
        
        # Keep only the specified views
        if type(views) is not list:
            views = [views]
        self.views = views
        
        self.csv["view"] = self.csv['Projection']
        self.csv = self.csv[self.csv["view"].isin(self.views)]

        # remove null stuff
        self.csv = self.csv[~self.csv["Labels"].isnull()]
        
        # remove missing files
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
                   "216840111366964012819207061112010306085429121_04-020-102.png"]
        self.csv = self.csv[~self.csv["ImageID"].isin(missing)]
        
        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()
            
        # filter out age < 10 (paper published 2019)
        self.csv = self.csv[(2019-self.csv.PatientBirth > 10)]
        
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    #print("mapping", syn)
                    mask |= self.csv["Labels"].str.contains(syn.lower())
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        self.pathologies[self.pathologies.index("Tube'")] = "Tube"
        
        ########## add consistent csv values
        
        # offset_day_int
        dt = pd.to_datetime(self.csv["StudyDate_DICOM"], format="%Y%m%d")
        self.csv["offset_day_int"] = dt.astype(np.int)// 10**9 // 86400
        
        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        imgid = self.csv['ImageID'].iloc[idx]
        img_path = os.path.join(self.imgpath,imgid)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)   
        
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]                    
                               
        if self.transform is not None:
            img = self.transform(img)
            
        if self.data_aug is not None:
            img = self.data_aug(img)

        return {"img":img, "lab":self.labels[idx], "idx":idx}

class CheX_Dataset(Dataset):
    """
    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.
    Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, 
    Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong, 
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, 
    Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng. https://arxiv.org/abs/1901.07031
    
    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/
    """
    def __init__(self, imgpath, csvpath, views=["PA"], transform=None, data_aug=None,
                 flat_dir=True, seed=0, unique_patients=True):

        super(CheX_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 255
        
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
        self.views = views
        
        # To list
        if type(self.views) is not list:
            views = [views]
            self.views = views
              
        self.csv["view"] = self.csv["Frontal/Lateral"] # Assign view column 
        self.csv.loc[(self.csv["view"] == "Frontal"), "view"] = self.csv["AP/PA"] # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv["view"] = self.csv["view"].replace({'Lateral': "L"}) # Rename Lateral with L  
        self.csv = self.csv[self.csv["view"].isin(self.views)] # Select the view 
         
        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(pat = '(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()
                   
        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
                
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        # make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan
        
        # rename pathologies
        self.pathologies = list(np.char.replace(self.pathologies, "Pleural Effusion", "Effusion"))
        
        ########## add consistent csv values
        
        # offset_day_int
        
        # patientid
        if 'train' in csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif 'valid' in csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplemented

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient","")
        self.csv["patientid"] = patientid
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        imgid = self.csv['Path'].iloc[idx]
        imgid = imgid.replace("CheXpert-v1.0-small/","")
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)      
        
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

        if self.transform is not None:
            img = self.transform(img)
            
        if self.data_aug is not None:
            img = self.data_aug(img)

        return {"img":img, "lab":self.labels[idx], "idx":idx}
    
class MIMIC_Dataset(Dataset):
    """
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY, Mark RG, Horng S. 
    MIMIC-CXR: A large publicly available database of labeled chest radiographs. 
    arXiv preprint arXiv:1901.07042. 2019 Jan 21.
    
    https://arxiv.org/abs/1901.07042
    
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """
    def __init__(self, imgpath, csvpath,metacsvpath, views=["PA"], transform=None, data_aug=None,
                 flat_dir=True, seed=0, unique_patients=True):

        super(MIMIC_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.MAXVAL = 255
        
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
        self.views = views
        if self.views:
            if type(views) is not list:
                views = [views]
            self.views = views

            self.csv["view"] = self.csv["ViewPosition"]
            self.csv = self.csv[self.csv["view"].isin(self.views)]

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()
                   
        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
                
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        # make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan
        
        # rename pathologies
        self.pathologies = np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        
        ########## add consistent csv values
        
        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]
        
        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])
        
        img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)      
        
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

        if self.transform is not None:
            img = self.transform(img)
            
        if self.data_aug is not None:
            img = self.data_aug(img)

        return {"img":img, "lab":self.labels[idx], "idx":idx}
    
class Openi_Dataset(Dataset):
    """
    OpenI 
    Dina Demner-Fushman, Marc D. Kohli, Marc B. Rosenman, Sonya E. Shooshan, Laritza
    Rodriguez, Sameer Antani, George R. Thoma, and Clement J. McDonald. Preparing a
    collection of radiology examinations for distribution and retrieval. Journal of the American
    Medical Informatics Association, 2016. doi: 10.1093/jamia/ocv080.
    
    Dataset website:
    https://openi.nlm.nih.gov/faq
    
    Download images:
    https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d
    """
    def __init__(self, imgpath, 
                 xmlpath=os.path.join(datapath, "NLMCXR_reports.tgz"), 
                 dicomcsv_path=os.path.join(datapath, "nlmcxr_dicom_metadata.csv.gz"),
                 tsnepacsv_path=os.path.join(datapath, "nlmcxr_tsne_pa.csv.gz"),
                 filter_pa=True,
                 transform=None, data_aug=None, 
                 nrows=None, seed=0,
                 pure_labels=False, unique_patients=True):

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
        self.xmlpath = xmlpath
        
        tarf = tarfile.open(xmlpath, 'r:gz')
        
        samples = []
        #for f in os.listdir(xmlpath):
        #   tree = xml.etree.ElementTree.parse(os.path.join(xmlpath, f))
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
        self.MAXVAL = 255  # Range [0 255]
            
        self.dicom_metadata = pd.read_csv(dicomcsv_path, index_col="imageid", low_memory=False)

        # merge in dicom metadata
        self.csv = self.csv.join(self.dicom_metadata, on="imageid")
            
        #filter only PA/AP view
        if filter_pa:
            tsne_pa = pd.read_csv(tsnepacsv_path, index_col="imageid")
            self.csv = self.csv.join(tsne_pa, on="imageid")

            self.csv = self.csv[self.csv["tsne-view"] == "PA"]
        
#         self.csv = self.csv[self.csv["View Position"] != "RL"]
#         self.csv = self.csv[self.csv["View Position"] != "LATERAL"]
#         self.csv = self.csv[self.csv["View Position"] != "LL"]
            
        if unique_patients:
            self.csv = self.csv.groupby("uid").first().reset_index()
            
        # Get our classes.        
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["labels_automatic"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    #print("mapping", syn)
                    mask |= self.csv["labels_automatic"].str.contains(syn.lower())
            self.labels.append(mask.values)
            
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        # rename pathologies
        self.pathologies = np.char.replace(self.pathologies, "Opacity", "Lung Opacity")
        self.pathologies = np.char.replace(self.pathologies, "Lesion", "Lung Lesion")
        
        ########## add consistent csv values
        
        # offset_day_int
        #self.csv["offset_day_int"] = 
        
        # patientid
        self.csv["patientid"] = self.csv["uid"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imageid = self.csv.iloc[idx].imageid
        img_path = os.path.join(self.imgpath,imageid + ".png")
        #print(img_path)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]                    
                               
        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)
            
        return {"img":img, "lab":self.labels[idx], "idx":idx}

class COVID19_Dataset(Dataset):
    """
    COVID-19 Image Data Collection: Prospective Predictions Are the Future
    Joseph Paul Cohen and Paul Morrison and Lan Dao and Karsten Roth and Tim Q Duong and Marzyeh Ghassemi
    arXiv:2006.11988, 2020
    
    COVID-19 image data collection, 
    Joseph Paul Cohen and Paul Morrison and Lan Dao
    arXiv:2003.11597, 2020

    Dataset: https://github.com/ieee8023/covid-chestxray-dataset
    
    Paper: https://arxiv.org/abs/2003.11597
    """
    
    def __init__(self, 
                 imgpath=os.path.join(thispath, "covid-chestxray-dataset", "images"), 
                 csvpath=os.path.join(thispath, "covid-chestxray-dataset", "metadata.csv"), 
                 semantic_masks_v7labs_lungs_path=os.path.join(datapath, "semantic_masks_v7labs_lungs.zip"),
                 views=["PA", "AP"],
                 transform=None, 
                 data_aug=None, 
                 nrows=None, 
                 seed=0,
                 pure_labels=False, 
                 unique_patients=True,
                 semantic_masks=False):

        super(COVID19_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.views = views
        self.semantic_masks = semantic_masks
        self.semantic_masks_v7labs_lungs_path = semantic_masks_v7labs_lungs_path
                
        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        self.MAXVAL = 255  # Range [0 255]

        # Keep only the frontal views.
        #idx_pa = self.csv["view"].isin(["PA", "AP", "AP Supine"])
        idx_pa = self.csv["view"].isin(self.views)
        self.csv = self.csv[idx_pa]
        
        #filter out in progress samples
        self.csv = self.csv[~(self.csv.finding == "todo")]
        self.csv = self.csv[~(self.csv.finding == "Unknown")]
        
        self.pathologies = self.csv.finding.str.split("/", expand=True).values.ravel()
        self.pathologies = self.pathologies[~pd.isnull(self.pathologies)]
        self.pathologies = sorted(np.unique(self.pathologies))
        
        
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["finding"].str.contains(pathology)
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        self.csv = self.csv.reset_index()
        
        if self.semantic_masks:
            temp = zipfile.ZipFile(self.semantic_masks_v7labs_lungs_path)
            self.semantic_masks_v7labs_lungs_namelist = temp.namelist()
            
        ########## add consistent csv values
        
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
        #print(img_path)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        sample["img"] = img[None, :, :]                    
                               
        transform_seed = np.random.randint(2147483647)
        
        if self.semantic_masks:
            sample["semantic_masks"] = self.get_semantic_mask_dict(imgid, sample["img"].shape)
            
        if self.transform is not None:
            random.seed(transform_seed)
            sample["img"] = self.transform(sample["img"])
            if self.semantic_masks:
                for i in sample["semantic_masks"].keys():
                    random.seed(transform_seed)
                    sample["semantic_masks"][i] = self.transform(sample["semantic_masks"][i])
  
        if self.data_aug is not None:
            random.seed(transform_seed)
            sample["img"] = self.data_aug(sample["img"])
            if self.semantic_masks:
                for i in sample["semantic_masks"].keys():
                    random.seed(transform_seed)
                    sample["semantic_masks"][i] = self.data_aug(sample["semantic_masks"][i])
            
        return sample
    
    def get_semantic_mask_dict(self, image_name, this_shape):
                
        archive_path = "semantic_masks_v7labs_lungs/" + image_name
        semantic_masks = {}
        if archive_path in self.semantic_masks_v7labs_lungs_namelist:
            with zipfile.ZipFile(self.semantic_masks_v7labs_lungs_path).open(archive_path) as file:
                mask = imageio.imread(file.read())
                
                mask = (mask == 255).astype(np.float)
                # reshape so image resizing works
                mask = mask[None, :, :] 

                semantic_masks["Lungs"] = mask

        return semantic_masks
    
class NLMTB_Dataset(Dataset):
    """
    National Library of Medicine Tuberculosis Datasets
    https://lhncbc.nlm.nih.gov/publication/pub9931
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/
    
    Jaeger S, Candemir S, Antani S, Wang YX, Lu PX, Thoma G. Two public chest X-ray 
    datasets for computer-aided screening of pulmonary diseases. Quant Imaging Med 
    Surg. 2014 Dec;4(6):475-7. doi: 10.3978/j.issn.2223-4292.2014.11.20. 
    PMID: 25525580; PMCID: PMC4256233.

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
                 views=["PA", "AP"]
        ):
        """
        Args:
            img_path (str): Path to `MontgomerySet` or `ChinaSet_AllFiles` folder
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

        #Label is the last digit on the simage filename
        self.csv["label"] = self.csv["fname"].apply(lambda x: int(x.split(".")[-2][-1]))

        self.labels = self.csv["label"].values.reshape(-1,1)
        self.pathologies = ["Tuberculosis"]
        self.views = views
        
        
        self.MAXVAL = 255

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.csv.iloc[idx]
        img_path = os.path.join(self.imgpath, "CXR_png", item["fname"])
        #print(img_path)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]                    
                               
        if self.transform is not None:
            img = self.transform(img)

        if self.data_aug is not None:
            img = self.data_aug(img)
            
        return {"img":img, "lab":self.labels[idx], "idx":idx}    

class SIIM_Pneumothorax_Dataset(Dataset):
    """
    https://academictorrents.com/details/6ef7c6d039e85152c4d0f31d83fa70edc4aba088
    https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
    
    "The data is comprised of images in DICOM format and annotations in the form of image IDs and run-length-encoded (RLE) masks. Some of the images contain instances of pneumothorax (collapsed lung), which are indicated by encoded binary masks in the annotations. Some training images have multiple annotations.
    Images without pneumothorax have a mask value of -1."
    """
    
    def __init__(self, 
                 imgpath, 
                 csvpath, 
                 transform=None, 
                 data_aug=None, 
                 seed=0,
                 unique_patients=True,
                 masks=False):

        super(SIIM_Pneumothorax_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.masks = masks
                
        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.MAXVAL = 255  # Range [0 255]
        
        self.pathologies = ["Pneumothorax"]
        
        self.labels = []
        self.labels.append(self.csv[" EncodedPixels"] != "-1")
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        self.csv = self.csv.reset_index()
        
        self.csv["has_masks"] = self.csv[" EncodedPixels"] != "-1"
        
        #to figure out the paths
        #TODO: make faster
        if not ("siim_file_map" in _cache_dict):
            file_map = {}
            for root, directories, files in os.walk(self.imgpath, followlinks=False):
                for filename in files:
                    filePath = os.path.join(root,filename)
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
        #print(img_path)
        img = pydicom.filereader.dcmread(img_path).pixel_array
        #img = imread(img_path)
        img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        sample["img"] = img[None, :, :]                    
                               
        transform_seed = np.random.randint(2147483647)
        
        if self.masks:
            sample["pathology_masks"] = self.get_pathology_mask_dict(imgid, sample["img"].shape[2])
            
        if self.transform is not None:
            random.seed(transform_seed)
            sample["img"] = self.transform(sample["img"])
            if self.masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.transform(sample["pathology_masks"][i])
  
        if self.data_aug is not None:
            random.seed(transform_seed)
            sample["img"] = self.data_aug(sample["img"])
            if self.masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.data_aug(sample["pathology_masks"][i])
            
        return sample
    
    def get_pathology_mask_dict(self, image_name, this_size):

        base_size = 1024
        images_with_masks = self.csv[np.logical_and(self.csv["ImageId"] == image_name, 
                                                    self.csv[" EncodedPixels"] != "-1")]
        path_mask = {}

        # from kaggle code
        def rle2mask(rle, width, height):
            mask= np.zeros(width* height)
            array = np.asarray([int(x) for x in rle.split()])
            starts = array[0::2]
            lengths = array[1::2]

            current_position = 0
            for index, start in enumerate(starts):
                current_position += start
                mask[current_position:current_position+lengths[index]] = 1
                current_position += lengths[index]

            return mask.reshape(width, height)
        
        if len(images_with_masks) > 0:
            # using a for loop so it is consistent with the other code
            for patho in ["Pneumothorax"]:
                mask = np.zeros([this_size,this_size])

                # don't add masks for labels we don't have
                if patho in self.pathologies:

                    for i in range(len(images_with_masks)):
                        row = images_with_masks.iloc[i]
                        mask = rle2mask(row[" EncodedPixels"],base_size,base_size)
                        mask = mask.T
                        mask = skimage.transform.resize(mask, (this_size, this_size), mode='constant', order=0)
                        mask = mask.round() #make 0,1

                # reshape so image resizing works
                mask = mask[None, :, :] 

                path_mask[self.pathologies.index(patho)] = mask
            
        return path_mask
    
class VinBrain_Dataset(Dataset):
    """
    Nguyen et al., VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations
    https://arxiv.org/abs/2012.15029
    
    https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
    """
    def __init__(self, imgpath, 
                 csvpath, 
                 views=None,
                 transform=None, 
                 data_aug=None,
                 seed=0,
                 pathology_masks=False):
        
        super(VinBrain_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        self.views = views
        
        self.pathologies = [ 'Aortic enlargement',
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
        
        self.normalize = normalize
        # Load data
        self.check_paths_exist()
        self.rawcsv = pd.read_csv(self.csvpath)
        self.csv = pd.DataFrame(self.rawcsv.groupby("image_id")["class_name"].apply(lambda x: "|".join(np.unique(x))))
        
        self.csv["has_masks"] = self.csv.class_name != "No finding"
        
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["class_name"].str.lower().str.contains(pathology.lower())
            if pathology in self.mapping:
                for syn in self.mapping[pathology]:
                    mask |= self.csv["class_name"].str.lower().str.contains(syn.lower())
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
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
        #print(img_path)
        from pydicom.pixel_data_handlers.util import apply_modality_lut
        dicom_obj = pydicom.filereader.dcmread(img_path)
        #print(dicom_obj)
        img = apply_modality_lut(dicom_obj.pixel_array, dicom_obj)
        img = pydicom.pixel_data_handlers.apply_windowing(img, dicom_obj)
        
        # Photometric Interpretation to see if the image needs to be inverted
        mode = dicom_obj[0x28, 0x04].value
        bitdepth = dicom_obj[0x28, 0x101].value
        
        # hack!
        if img.max() < 256:
            bitdepth = 8
            
        if mode == "MONOCHROME1":
            img = -1*img + 2**float(bitdepth)
        elif mode == "MONOCHROME2":
            pass
        else:
            raise Exception("Unknown Photometric Interpretation mode")
            
        if self.normalize:
            img = normalize(img, 2**float(bitdepth))

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        sample["img"] = img[None, :, :]
        
        transform_seed = np.random.randint(2147483647)
        
        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape)
            
        if self.transform is not None:
            random.seed(transform_seed)
            sample["img"] = self.transform(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.transform(sample["pathology_masks"][i])
  
        if self.data_aug is not None:
            random.seed(transform_seed)
            sample["img"] = self.data_aug(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.data_aug(sample["pathology_masks"][i])
            
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
    """
    This dataset loads the Stonybrook 
    Radiographic Assessment of Lung Opacity Score Dataset
    
    https://doi.org/10.5281/zenodo.4633999
    
    Citation will be set soon.

    """
    
    def __init__(self, 
                 imgpath, # path to CXR_images_scored
                 csvpath, # path to ralo-dataset-metadata.csv
                 transform=None, 
                 data_aug=None, 
                 seed=0):

        super(StonyBrookCOVID_Dataset, self).__init__()
        
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
                
        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, skiprows=1)
        self.MAXVAL = 255  # Range [0 255]

        self.pathologies = ["Geographic Extent","Lung Opacity"]
        
        self.csv["Geographic Extent"] = (self.csv["Total GEOGRAPHIC"] + self.csv["Total GEOGRAPHIC.1"])/2
        self.csv["Lung Opacity"] = (self.csv["Total OPACITY"] + self.csv["Total OPACITY.1"])/2
        
        self.labels = []
        self.labels.append(self.csv["Geographic Extent"])
        self.labels.append(self.csv["Lung Opacity"])
        
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        ########## add consistent csv values
        
        # offset_day_int
        
        date_col = self.csv["Exam_DateTime"].str.split("_",expand=True)[0]
        dt = pd.to_datetime(date_col, format="%Y%m%d")
        self.csv["offset_day_int"] = dt.astype(np.int)// 10**9 // 86400
        
        # patientid
        self.csv["patientid"] = self.csv["Subject_ID"].astype(str)

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        
        img_path = os.path.join(self.imgpath, str(idx) + ".jpg")
        #print(img_path)
        img = imread(img_path)
        img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        sample["img"] = img[None, :, :]                    
                               
        transform_seed = np.random.randint(2147483647)
            
        if self.transform is not None:
            random.seed(transform_seed)
            sample["img"] = self.transform(sample["img"])
  
        if self.data_aug is not None:
            random.seed(transform_seed)
            sample["img"] = self.data_aug(sample["img"])
            
        return sample 
    
class ObjectCXR_Dataset(Dataset):
    """
    https://jfhealthcare.github.io/object-CXR/

    https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5
    """
    def __init__(self, imgzippath, 
                 csvpath, 
                 transform=None, 
                 data_aug=None,
                 seed=0):

        super(ObjectCXR_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgzippath = imgzippath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug
        self.views = []
        self.MAXVAL = 255  # Range [0 255]
        self.pathologies = ['Foreign Object']

        # Load data
        self.csv = pd.read_csv(self.csvpath)

        self.labels = []
        self.labels.append(~self.csv["annotation"].isnull())
        self.labels = np.asarray(self.labels).T
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
            img = imageio.imread(file.read())
            img = normalize(img, self.MAXVAL)

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        sample["img"] = img[None, :, :]

        transform_seed = np.random.randint(2147483647)

        if self.transform is not None:
            random.seed(transform_seed)
            sample["img"] = self.transform(sample["img"])

        if self.data_aug is not None:
            random.seed(transform_seed)
            sample["img"] = self.data_aug(sample["img"])

        return sample
    
    
class ToPILImage(object):
    def __init__(self):
        self.to_pil = transforms.ToPILImage(mode="F")

    def __call__(self, x):
        return(self.to_pil(x[0]))


class XRayResizer(object):
    def __init__(self, size, engine="skimage"):
        self.size = size
        self.engine = engine
        if 'cv2' in sys.modules:
            print("Setting XRayResizer engine to cv2 could increase performance.")

    def __call__(self, img):
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), mode='constant', preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            import cv2 # pip install opencv-python
            return cv2.resize(img[0,:,:], 
                              (self.size, self.size), 
                              interpolation = cv2.INTER_AREA
                             ).reshape(1,self.size,self.size).astype(np.float32)
        else:
            raise Exception("Unknown engine, Must be skimage (default) or cv2.")

class XRayCenterCrop(object):
    
    def crop_center(self, img):
        _, y, x = img.shape
        crop_size = np.min([y,x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]
    
    def __call__(self, img):
        return self.crop_center(img)

class CovariateDataset(Dataset):
    """
    Dataset which will correlate the dataset with a specific label.
    
    Viviano et al. Saliency is a Possible Red Herring When Diagnosing Poor Generalization
    https://arxiv.org/abs/1910.00199
    """
    def __init__(self, 
                 d1, d1_target,
                 d2, d2_target,
                 ratio=0.5, mode="train",
                 seed=0, nsamples=None,
                 splits=[0.5, 0.25, 0.25],
                 verbose=False):
        
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

        idx_sick = all_labels==1
        n_per_category = np.min([sum(idx_sick[all_site==0]),
                                 sum(idx_sick[all_site==1]),
                                 sum(~idx_sick[all_site==0]),
                                 sum(~idx_sick[all_site==1])])

        if verbose:
            print("n_per_category={}".format(n_per_category))

        all_0_neg = all_idx[np.where((all_site==0) & (all_labels==0))]
        all_0_neg = np.random.choice(all_0_neg, n_per_category, replace=False)
        all_0_pos = all_idx[np.where((all_site==0) & (all_labels==1))]
        all_0_pos = np.random.choice(all_0_pos, n_per_category, replace=False)
        all_1_neg = all_idx[np.where((all_site==1) & (all_labels==0))]
        all_1_neg = np.random.choice(all_1_neg, n_per_category, replace=False)
        all_1_pos = all_idx[np.where((all_site==1) & (all_labels==1))]
        all_1_pos = np.random.choice(all_1_pos, n_per_category, replace=False)

        # TRAIN
        train_0_neg = np.random.choice(
            all_0_neg, int(n_per_category*ratio*splits[0]*2), replace=False)
        train_0_pos = np.random.choice(
            all_0_pos, int(n_per_category*(1-ratio)*splits[0]*2), replace=False)
        train_1_neg = np.random.choice(
            all_1_neg, int(n_per_category*(1-ratio)*splits[0]*2), replace=False)
        train_1_pos = np.random.choice(
            all_1_pos, int(n_per_category*ratio*splits[0]*2), replace=False)

        # REDUCE POST-TRAIN
        all_0_neg = np.setdiff1d(all_0_neg, train_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, train_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, train_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, train_1_pos)

        if verbose:
            print("TRAIN: neg={}, pos={}".format(len(train_0_neg)+len(train_1_neg),
                                                 len(train_0_pos)+len(train_1_pos)))

        # VALID
        valid_0_neg = np.random.choice(
            all_0_neg, int(n_per_category*(1-ratio)*splits[1]*2), replace=False)
        valid_0_pos = np.random.choice(
            all_0_pos, int(n_per_category*ratio*splits[1]*2), replace=False)
        valid_1_neg = np.random.choice(
            all_1_neg, int(n_per_category*ratio*splits[1]*2), replace=False)
        valid_1_pos = np.random.choice(
            all_1_pos, int(n_per_category*(1-ratio)*splits[1]*2), replace=False)

        # REDUCE POST-VALID
        all_0_neg = np.setdiff1d(all_0_neg, valid_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, valid_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, valid_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, valid_1_pos)

        if verbose:
            print("VALID: neg={}, pos={}".format(len(valid_0_neg)+len(valid_1_neg),
                                                 len(valid_0_pos)+len(valid_1_pos)))

        # TEST
        test_0_neg = all_0_neg
        test_0_pos = all_0_pos
        test_1_neg = all_1_neg
        test_1_pos = all_1_pos

        if verbose:
            print("TEST: neg={}, pos={}".format(len(test_0_neg)+len(test_1_neg),
                                                len(test_0_pos)+len(test_1_pos)))

        def _reduce_nsamples(nsamples, a, b, c, d):
            if nsamples:
                a = a[:int(np.floor(nsamples/4))]
                b = b[:int(np.ceil(nsamples/4))]
                c = c[:int(np.ceil(nsamples/4))]
                d = d[:int(np.floor(nsamples/4))]

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
        self.labels = all_labels[self.select_idx].reshape(-1,1)
        self.site = all_site[self.select_idx]
        
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
        img = sample["img"]
        
        # replace the labels with the specific label we focus on
        sample["lab-old"] = sample["lab"]
        sample["lab"] = self.labels[idx]
        
        sample["site"] = self.site[idx]

        return sample
