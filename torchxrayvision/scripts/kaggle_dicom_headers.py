import pandas as pd
import pydicom
import os
from tqdm import tqdm

a = pd.DataFrame()
path = "/lustre03/project/6008064/jpcohen/kaggle-pneumonia/stage_2_train_images/"
for f in tqdm(os.listdir(path)):
    dataset = pydicom.dcmread(path+f)
    ent = {}
    for key in ["PatientID", 
                "Modality", 
                "ConversionType", 
                "PatientAge", 
                "PatientSex", 
                "PatientOrientation", 
                "BodyPartExamined", 
                "SeriesNumber", 
                "InstanceNumber", 
                "SamplesPerPixel", 
                "ViewPosition"]:
        ent[key] = eval("dataset." + key)
    ent['PixelSpacing'] = dataset.PixelSpacing[0] 
    ent['Filename'] = f
    a = a.append(ent, ignore_index=True)
a.to_csv("kaggle_stage_2_train_images_dicom_headers.csv.gz")
