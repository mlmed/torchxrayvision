# Finds the matches between the RSNA pneumonia dataset and the NIH dataset
# performs a match by first matching the metadata (patient age, sex, view pos)
# then calculating mean squared error.
# The RSNA data has been lossy compressed, therefore the match is not perfect,
# however there is a distinct difference between matching cases (MSE usually
# around 2, and non matching cases, MSE > 30, usually much greater)
# A cutoff of 10 is effective
import glob
import numpy as np
import zipfile
from io import BytesIO
from PIL import Image
import time
import pathlib
import pickle
import os
import tqdm
import pydicom
import pandas as pd

output_file="kagglematches.pkl"
cxr_14_image_path='/tmp/datapath/chest_xray/images'
cxr_14_csv_path='/tmp/datapath/chest_xray/Data_Entry_2017.csv'
cxr14_files=glob.glob(os.path.join(cxr_14_image_path,'*.png'))
print("There are",len(cxr14_files),"files")
cxr14_metadata=pd.read_csv(cxr_14_csv_path)

cxr14_orig=dict()
print("Organize cxr14 metadata")
for _,row in tqdm.tqdm(cxr14_metadata.iterrows()):
    cxr14_orig[row["Image Index"]]=row.to_dict()


rsna_zip='/tmp/datapath/rsna-pneumonia/rsna-pneumonia-detection-challenge.zip'
with zipfile.ZipFile(rsna_zip) as zf:
    rsna_train_labels=pd.read_csv(BytesIO(zf.read('stage_2_train_labels.csv')))
    rsna_detailed_class_info=pd.read_csv(BytesIO(zf.read('stage_2_detailed_class_info.csv')))

# This uses a LOT of ram, alternately you can just load the images when needed below
#cxr14data=dict()
#print("Loading CXR14 images")
#for f in tqdm.tqdm(cxr14_files):
#    d=np.array(Image.open(f).convert(mode='L'))
#    name=pathlib.Path(f).stem
#    cxr14data[name]=d

rsna_info=dict()
print("Making rsna image index")
for i,(_,row) in tqdm.tqdm(enumerate(rsna_detailed_class_info.iterrows())):
    if row['patientId'] not in rsna_info:
        rsna_info[row['patientId']]={'info':[]}
    rsna_info[row['patientId']]['info'].append(row['class'])

print("Performing file matching")

with zipfile.ZipFile(rsna_zip) as zf:
    nitems=len(rsna_info.items())
    pbar_outer=tqdm.tqdm(enumerate(rsna_info.items()))
    for i,(k,v) in pbar_outer:
        pbar_outer.set_description(k)
        f="stage_2_train_images/%s.dcm"%k
        fd=pydicom.filereader.dcmread(BytesIO(zf.read(f)))
        rsna_img=fd.pixel_array
        v['mse']=np.inf
        rsna_sex=fd.PatientSex
        # The rsna age is not actually a proper DICOM Age string, and loses the year/month
        # qualifier  A handful of cases were actually in months not years.  Strip to the int
        # and compare that.
        rsna_age=int(fd.PatientAge)
        rsna_vp=fd.ViewPosition
        pbar_inner= tqdm.tqdm(cxr14_orig.items())
        pbar_inner.set_description("%s MSE: %.2f"%("no file",np.inf))
        for kk,vv in pbar_inner:
            if (vv['Patient Gender']==rsna_sex
                and int(vv['Patient Age'][:-1])== rsna_age
                and vv['View Position']== rsna_vp
              ):
                cxr14_img=np.array(Image.open(os.path.join(cxr_14_image_path,kk)).convert(mode='L'))
                # use the following if you preload the images
                #cxr14_img=cxr14data[kk.split('.')[0]]
                mse=np.mean(np.square(cxr14_img-rsna_img))
                if mse<v['mse']:
                    v['mse']=mse
                    v['cxr_match']=kk
                    pbar_inner.set_description("%s MSE: %.2f"%(f,mse))
with open(outputfile,'wb') as fd:
    pickle.dump(rsna_info,fd)


