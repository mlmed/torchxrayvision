import numpy as np
import pdb
import tarfile
import pandas as pd
from PIL import Image
import random
import argparse
from pathlib import Path
import os

from random_data import write_random_images, gen_int, gen_hex, random_pred, random_preds

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
        subject_id = gen_int(8)
        study_id = gen_int(8)
        meta_row = {
            "dicom_id":"-".join([gen_hex(8) for i in range(4)]),
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

        csv_row = {
            "subject_id":subject_id,
            "study_id":study_id,
        }
        csv_row.update(random_preds())
        return meta_row, csv_row

    meta_rows, csv_rows = show(list(zip(*show([generate_random_row(dimensions) for i in range(n)]))))

    return pd.DataFrame(meta_rows), pd.DataFrame(csv_rows)



def generate_test_images(random_metadata, extracted, tarname, zipname, folder_of_zip_name, folder_of_tar_gz_name, dimensions):
    paths = []
    for _, row in random_metadata.iterrows():
        subjectid = row["subject_id"]
        studyid = row["study_id"]
        dicom_id = row["dicom_id"]
        img_fname = os.path.join("p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        paths.append(Path("files")/img_fname)
    write_random_images(paths, extracted, tarname, zipname, folder_of_zip_name, folder_of_tar_gz_name, dimensions)

def generate_test_data(n, directory, dimensions=(224, 224), tarname=None, zipname=None, folder_of_zip_name=None, folder_of_tar_gz_name = None, extracted=None):
    directory = Path(directory)
    if tarname is None:
        tarname = directory/"images-224.tar"
    if zipname is None:
        zipname = directory/"images-224.zip"
    if extracted is None:
        extracted = directory/"images-224"
    if folder_of_zip_name is None:
        folder_of_zip_name = directory/"images-224-zips"
    if folder_of_tar_gz_name is None:
        folder_of_tar_gz_name = directory/"images-224-tgzs"
    random_metadata, random_csvdata = generate_random_metadata(
        n,
        dimensions
    )
    generate_test_images(random_metadata, extracted, tarname, zipname, folder_of_zip_name, folder_of_tar_gz_name, dimensions)
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
