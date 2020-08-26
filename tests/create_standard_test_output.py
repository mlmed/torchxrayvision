import torchxrayvision as xrv
import os

n=10

def create_standard_test_output(dataset, name):
    dataset.csv[:n].to_csv(name)

create_standard_test_output(
    xrv.datasets.NIH_Dataset(imgpath="."),
    "nih.csv"
)

create_standard_test_output(
    xrv.datasets.PC_Dataset(imgpath="."),
    "pc.csv"
)

create_standard_test_output(
    xrv.datasets.Openi_Dataset(imgpath="."),
    "openi.csv"
)

#NLMTB Shenzen
create_standard_test_output(
    xrv.datasets.NLMTB_Dataset(imgpath=os.path.expanduser("~/ChinaSet_AllFiles.zip")),
    "shenzen.csv"
)

#NLMTB Montgomery
create_standard_test_output(
    xrv.datasets.NLMTB_Dataset(imgpath=os.path.expanduser("~/NLM-MontgomeryCXRSet.zip")),
    "montgomery.csv"
)

#RSNA train
create_standard_test_output(
    xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="."), #csvpath="kaggle_stage_2_train_images_dicom_headers.csv.gz")
    "rsna_train.csv"
)
