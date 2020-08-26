from random_data import random_preds
import numpy as np
import pandas as pd
import random

def probability(fraction):
    gran = 100
    return np.random.randint(0, gran) < gran * fraction

def random_age():
    return random.randint(18, 100) #Assume adult

def gen_random_study():
    views = []
    if probability(.7):
        views.append({
            "Path":"view1_frontal.jpg",
            "AP/PA":np.random.choice(["AP","PA"]),
            "Frontal/Lateral":"Frontal"
        })
    if probability(.3):
        views.append({
            "Path":"view1_lateral.jpg",
            "AP/PA":"",
            "Frontal/Lateral":"Lateral"
        })
    return views

def gen_random_data():
    views = []
    patient_idx = 0
    while True:
        n_studies = random.randint(1, 4)
        sex = random.choice(["Female","Male"])
        age = random_age()
        for study_idx in range(n_studies):
            for radiograph in gen_random_study():
                path_format = (
                    "CheXpert-v1.0-small/train/patient{}/study{}/{}"
                )
                path = path_format.format(
                    str(patient_idx).zfill(5),
                    str(study_idx),
                    radiograph["Path"]
                )
                radiograph.update({
                    "Path":path,
                    "Sex":sex,
                    "Age":age,
                })
                radiograph.update(random_preds())
                yield radiograph
        patient_idx += 1

#Path,Sex,Age,Frontal/Lateral,AP/PA,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices
#CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg,Female,68,Frontal,AP,1.0,,,,,,,,,0.0,,,,1.0

def gen_random_rows(nrows):
    rows = []
    random_data_source = gen_random_data()
    for i in range(nrows):
         rows.append(next(random_data_source))
    return pd.DataFrame(rows)



gen_random_rows(10).to_csv("test_chexpert_data.csv")
