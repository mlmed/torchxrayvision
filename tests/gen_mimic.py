import numpy as np
import pdb
import tarfile
import pandas as pd
from PIL import Image
import random
import argparse
from pathlib import Path
import os

def show(x):
    print(x)
    return x

mimic_metadata_filename = "mimic-cxr-2.0.0-metadata.csv"
mimic_csvdata_filename = "mimic-cxr-2.0.0-negbio.csv"

def generate_random_metadata(n, dimensions):
    columns = "dicom_id,subject_id,study_id,PerformedProcedureStepDescription,ViewPosition,Rows,Columns,StudyDate,StudyTime,ProcedureCodeSequence_CodeMeaning,ViewCodeSequence_CodeMeaning,PatientOrientationCodeSequence_CodeMeaning".split(",")
    performed_procedure_step_descriptions = {
        "CHEST (PA AND LAT)":{
             "n_views":2,
             "view_position":["LATERAL", "PA"],
             "procedure_code_meaning":"CHEST (PA AND LAT)",
             "view_code_meaning":["lateral", "postero-anterior"],
             "orientation_code_meaning":["Erect","Recumbent"]
        }
    }
    def hex(n):
        hex_chars = list("0123456789abcdef")
        return "".join(np.random.choice(hex_chars,n))

    def int(n):
        int_chars = list("0123456789abcdef")
        return "".join(np.random.choice(int_chars,n))

    def generate_random_row(dimensions):
        performed_procedure_step_description = random.choice(
            list(performed_procedure_step_descriptions)
        )
        procedure = performed_procedure_step_descriptions[performed_procedure_step_description]
        n_views = procedure["n_views"]
        view_index = random.randint(0, n_views - 1)
        view_position = procedure["view_position"][view_index]
        procedure_code_meaning = procedure["procedure_code_meaning"]
        view_code_meaning = procedure["view_code_meaning"][view_index]
        #Currently unsure how/if view codes are mapped to orientations
        orientation_code_meaning = random.choice(procedure["orientation_code_meaning"])
        subject_id = int(8)
        study_id = int(8)
        meta_row = {
            "dicom_id":"-".join([hex(8) for i in range(4)]),
            "subject_id":subject_id,
            "study_id":study_id,
            "PerformedProcedureStepDescription":performed_procedure_step_description,
            "ViewPosition":view_position,
            "Rows":dimensions[0],
            "Columns":dimensions[1],
            "StudyDate":0,
            "StudyTime":0,
            "ProcedureCodeSequence_CodeMeaning":procedure_code_meaning,
            "ViewCodeSequence_CodeMeaning":view_code_meaning,
            "PatientOrientationCodeSequence_CodeMeaning":orientation_code_meaning
        }

        def random_pred():
            return random.choice(["1.0","-1.0","0.0",""])

        csv_row = {
            "subject_id":subject_id,
            "study_id":study_id,
            "Atelectasis":random_pred(),
            "Cardiomegaly":random_pred(),
            "Consolidation":random_pred(),
            "Edema":random_pred(),
            "Enlarged Cardiomediastinum":random_pred(),
            "Fracture":random_pred(),
            "Lung Lesion":random_pred(),
            "Lung Opacity":random_pred(),
            "No Finding":random_pred(),
            "Pleural Effusion":random_pred(),
            "Pleural Other":random_pred(),
            "Pneumonia":random_pred(),
            "Pneumothorax":random_pred(),
            "Support Devices":random_pred()
        }
        return meta_row, csv_row

    meta_rows, csv_rows = show(list(zip(*show([generate_random_row(dimensions) for i in range(n)]))))

    return pd.DataFrame(meta_rows), pd.DataFrame(csv_rows)


def generate_random_image(dimensions):
    return Image.fromarray(np.random.random(dimensions)).convert("L")

def generate_test_images(random_metadata, extracted, tarname, dimensions):
    for _, row in random_metadata.iterrows():
        subjectid = row["subject_id"]
        studyid = row["study_id"]
        dicom_id = row["dicom_id"]
        img_fname = os.path.join("p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        print(type(extracted))
        img_path = extracted/"files"/img_fname
        print(img_path)
        os.makedirs(os.path.dirname(img_path))
        generate_random_image(dimensions).save(img_path)
    tarred = tarfile.TarFile.open(tarname, "w")
    tarred.add(extracted)

def generate_test_data(n, directory, dimensions=(224, 224), tarname=None, extracted=None):
    directory = Path(directory)
    if tarname is None:
        tarname = directory/"images-224.tar"
    if extracted is None:
        extracted = directory/"images-224"
    random_metadata, random_csvdata = generate_random_metadata(
        n,
        dimensions
    )
    generate_test_images(random_metadata, extracted, tarname, dimensions)
    random_metadata.to_csv(
        directory/mimic_metadata_filename,
        index=False
    )
    random_metadata.to_csv(
        directory/(mimic_metadata_filename+".gz"),
        compression="gzip",
        index=False
    )
    random_csvdata.to_csv(
        directory/mimic_csvdata_filename,
        index=False
    )
    random_csvdata.to_csv(
        directory/(mimic_csvdata_filename+".gz"),
        compression="gzip",
        index=False
    )

#./images-224/files/p17/p17387118/s56770356/b983f94c-b77ad35d-8a4aa372-2faf6503-5ec94835.jpg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n")
    parser.add_argument("directory")
    parser.add_argument("x")
    parser.add_argument("y")
    parser.add_argument("tarfile", default=None, nargs="?")
    parser.add_argument("extracted", default=None, nargs="?")
    args = parser.parse_args()
    generate_test_data(
        n=int(args.n),
        directory = args.directory,
        dimensions = (int(args.x), int(args.y)),
        tarname = args.tarfile,
        extracted = args.extracted
    )
