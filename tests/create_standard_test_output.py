import torchxrayvision as xrv
import os

n=10

def create_standard_test_output(dataset, name, columns = None):
    minimal_dataset = dataset.csv[:n]
    if columns is not None:
        minimal_dataset = minimal_dataset[columns]
    minimal_dataset.to_csv(name)

print("nih")
create_standard_test_output(
    xrv.datasets.NIH_Dataset(imgpath="."),
    "nih.csv"
)

print("pc")
create_standard_test_output(
    xrv.datasets.PC_Dataset(imgpath="."),
    "pc.csv"
)

print("openi")
openi = xrv.datasets.Openi_Dataset(imgpath=".")
create_standard_test_output(
    openi,
    "openi.csv",
    columns=list(openi.dicom_metadata.columns) + ["imageid"]
)

#NLMTB Shenzen
print("shenzen")
create_standard_test_output(
    xrv.datasets.NLMTB_Dataset(imgpath=os.path.expanduser("~/ChinaSet_AllFiles.zip")),
    "shenzen.csv"
)

#NLMTB Montgomery
print("montgomery")
create_standard_test_output(
    xrv.datasets.NLMTB_Dataset(imgpath=os.path.expanduser("~/NLM-MontgomeryCXRSet.zip")),
    "montgomery.csv"
)

#RSNA train
print("rsna (just train data)")
rsna = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath=".")
create_standard_test_output(
    xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="."),
    "rsna_train.csv",
    columns=list(rsna.raw_csv.columns) + ["]
)
