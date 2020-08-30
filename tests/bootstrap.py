#bash generate_all.sh
from create_standard_test_output import create_standard_test_output
from gen_mimic import generate_mimic_test_data
from gen_chexpert import gen_chexpert
import torchxrayvision as xrv
from generate_test_data import generate_test_data
import pandas as pd

n = 10

def create_csv_files(n):
    print("nih")
    create_standard_test_output(
        xrv.datasets.NIH_Dataset(imgpath="."),
        "nih.csv",
        n=n
    )

    print("pc")
    create_standard_test_output(
        xrv.datasets.PC_Dataset(imgpath="."),
        "pc.csv",
        n=n
    )

    print("openi")
    openi = xrv.datasets.Openi_Dataset(imgpath=".")
    create_standard_test_output(
        openi,
        "openi.csv",
        columns=list(openi.dicom_metadata.columns) + ["imageid"],
        n=n
    )

    #RSNA train
    print("rsna (just train data)")
    rsna = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath=".")
    create_standard_test_output(
        xrv.datasets.RSNA_Pneumonia_Dataset(imgpath="."),
        "rsna_train.csv",
        columns=list(rsna.raw_csv.columns) + ["patientId"],
        n=n
    )

    #Chexpert
    gen_chexpert(n, "test_chexpert_data.csv")

    #No data is generated for the COVID19, NLM_TB (Shenzen) or
    #NLM_TB (Montgomery) datasets.

def create_images():

    #python3 generate_test_data.py pc.csv ImageID 2 2 PC_test_data
    generate_test_data(
        pd.read_csv("pc.csv"),
        "ImageID",
        (2, 2),
        "PC_test_data",
        "",
        "."
    )

    #python3 generate_test_data.py nih.csv "Image Index" 2 2 NIH_test_data
    generate_test_data(
         pd.read_csv("nih.csv"),
        "Image Index",
        (2, 2),
        "NIH_test_data",
        "",
        "."
    )

    #python3 generate_test_data.py openi.csv imageid 2 2 Openi_test_data .png
    generate_test_data(
        pd.read_csv("openi.csv"),
        "imageid",
        (2, 2),
        "Openi_test_data",
        ".png",
        "."
    )

    #python3 generate_test_data.py shenzen.csv fname 2 2 Shenzen_test_data --subfolder CXR_png
    generate_test_data(
        pd.read_csv("shenzen.csv"),
        "fname",
        (2, 2),
        "Shenzen_test_data",
        "",
        "CXR_png"
    )

    #python3 generate_test_data.py montgomery.csv fname 2 2 Montgomery_test_data --subfolder CXR_png
    generate_test_data(
        pd.read_csv("montgomery.csv"),
        "fname",
        (2, 2),
        "Montgomery_test_data",
        "",
        "CXR_png"
    )

    #python3 generate_test_data.py rsna_train.csv patientId 2 2 RSNA_test_data_jpg .jpg --subfolder stage_2_train_images
    generate_test_data(
        pd.read_csv("rsna_train.csv"),
        "patientId",
        (2, 2),
        "RSNA_test_data_jpg",
        ".jpg",
        "stage_2_train_images"
    )

    #python3 generate_test_data.py rsna_train.csv patientId 2 2 RSNA_test_data_dcm .dcm --subfolder stage_2_train_images
    generate_test_data(
        pd.read_csv("rsna_train.csv"),
        "patientId",
        (2, 2),
        "RSNA_test_data_dcm",
        ".dcm",
        "stage_2_train_images"
    )

    #python3 generate_test_data.py test_chexpert_data.csv Path 2 2 CheXpert_test_data
    generate_test_data(
        pd.read_csv("test_chexpert_data.csv"),
        "Path",
        (2, 2),
        "CheXpert_test_data",
        "",
        "."
    )

    #python3 generate_test_data.py test_covid_data.csv filename 2 2 COVID_test_data --subfolder images
    generate_test_data(
        pd.read_csv("test_covid_data.csv"),
        "filename",
        (2, 2),
        "COVID_test_data",
        "",
        "."
    )

def bootstrap_test_cases(n):
    create_csv_files(n)
    create_images()
    #MIMIC is handled differently due to its unique structure
    generate_mimic_test_data(
        n,
        directory = "gen_mimic",
        dimensions = (2, 2)
    )

if __name__ == "__main__":
    bootstrap_test_cases(10)
