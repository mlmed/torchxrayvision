import pytest
import pickle
import torchxrayvision as xrv
import os
import torch
from pathlib import Path

dataset_classes = [xrv.datasets.NIH_Dataset,
                   xrv.datasets.Openi_Dataset,
                   xrv.datasets.NIH_Google_Dataset,
                   xrv.datasets.PC_Dataset]

def test_dataloader_basic():
    for dataset_class in dataset_classes:
        c = dataset_class(imgpath=".")
        c.image_interface.close()

def test_dataloader_merging():
    
    datasets = []
    for dataset_class in dataset_classes:
        print(dataset_class)
        dataset = dataset_class(imgpath=".")
        datasets.append(dataset)
    
    for dataset in datasets:
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, dataset)
        
    dd = xrv.datasets.Merge_Dataset(datasets)

# test that we catch incorrect pathology alignment
def test_dataloader_merging_incorrect_alignment():
    with pytest.raises(Exception) as excinfo:
    
        d_nih = xrv.datasets.NIH_Dataset(imgpath=".")
        d_pc = xrv.datasets.PC_Dataset(imgpath=".")

        dd = xrv.datasets.Merge_Dataset([d_nih, d_pc])
        
    assert "incorrect pathology alignment" in str(excinfo.value)
    
    
    with pytest.raises(Exception) as excinfo:
    
        d_nih = xrv.datasets.NIH_Dataset(imgpath=".")
        d_pc = xrv.datasets.PC_Dataset(imgpath=".")
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d_nih)
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies[:-1], d_pc)

        dd = xrv.datasets.Merge_Dataset([d_nih, d_pc])
        
    assert "incorrect pathology alignment" in str(excinfo.value)
    

def all_equal(items):
    if len(items) == 1:
        return True
    return all([item == items[0] for item in items[1:]])

def _test_opening_formats(dataset_class, imgpaths, n=10, **kwargs):
    sources = []
    for imgpath in imgpaths:
        dataset = dataset_class(imgpath=imgpath, **kwargs)
        #Add serial version
        sources.append(dataset)
    #Assert all items are the same in serial version
    for i, one_item_from_each in enumerate(zip(*sources)):
        print(i)
        if i >= n - 1:
            break
        assert all_equal([pickle.dumps(item) for item in one_item_from_each])
    #Try loading each in a parallel way
    #for source in sources:
    #     source.csv = source.csv.iloc[:10]
    #     source.labels = source.labels[:10]
    #     dataset = torch.utils.data.DataLoader(
    #                 source,
    #                 batch_size=10,
    #                 shuffle=False,
    #                 num_workers=8,
    #                 pin_memory=False
    #     )
    #     for i, _ in enumerate(dataset):
    #         if i >= n - 1:
    #             break
    for source in sources:
        source.image_interface.close()

def test_mimic_formats():
    _test_opening_formats(
        xrv.datasets.MIMIC_Dataset,
        imgpaths=[
            "tests/gen_mimic/images-224/files",
            "tests/gen_mimic/images-224.tar",
            "tests/gen_mimic/images-224.zip",
            "tests/gen_mimic/images-224-zips_1",
            "tests/gen_mimic/images-224-zips_2",
            "tests/gen_mimic/images-224-tgzs_1",
            "tests/gen_mimic/images-224-tgzs_2"
        ],
        csvpath="tests/gen_mimic/mimic-cxr-2.0.0-negbio.csv",
        metacsvpath="tests/gen_mimic/mimic-cxr-2.0.0-metadata.csv"
    )

def test_nih_formats():
    _test_opening_formats(
        xrv.datasets.NIH_Dataset,
        imgpaths=[
            "tests/NIH_test_data/folder",
            "tests/NIH_test_data/tar.tar",
            "tests/NIH_test_data/zip.zip",
            "tests/NIH_test_data/zipped_1",
            "tests/NIH_test_data/zipped_2",
            "tests/NIH_test_data/tgz_1",
            "tests/NIH_test_data/tgz_2"
        ],
        csvpath="tests/nih.csv"
    )

def test_pc_formats():
    _test_opening_formats(
        xrv.datasets.PC_Dataset,
        imgpaths=[
            "tests/PC_test_data/folder",
            "tests/PC_test_data/tar.tar",
            "tests/PC_test_data/zip.zip",
            "tests/PC_test_data/zipped_1",
            "tests/PC_test_data/zipped_2",
            "tests/PC_test_data/tgz_1",
            "tests/PC_test_data/tgz_2"
        ],
        csvpath="tests/pc.csv"
    )

def test_shenzen_formats():
    _test_opening_formats(
        xrv.datasets.NLMTB_Dataset,
        imgpaths=[
            "tests/Shenzen_test_data/folder",
            "tests/Shenzen_test_data/tar.tar",
            "tests/Shenzen_test_data/zip.zip",
            "tests/Shenzen_test_data/zipped_1",
            "tests/Shenzen_test_data/zipped_2",
            "tests/Shenzen_test_data/tgz_1",
            "tests/Shenzen_test_data/tgz_2",
        ]
    )

def test_montgomery_formats():
    _test_opening_formats(
        xrv.datasets.NLMTB_Dataset,
        imgpaths=[
            "tests/Montgomery_test_data/folder",
            "tests/Montgomery_test_data/tar.tar",
            "tests/Montgomery_test_data/zip.zip",
            "tests/Montgomery_test_data/zipped_1",
            "tests/Montgomery_test_data/zipped_2",
            "tests/Montgomery_test_data/tgz_1",
            "tests/Montgomery_test_data/tgz_2"
        ]
    )

def test_rsna_jpg_formats():
    _test_opening_formats(
        xrv.datasets.RSNA_Pneumonia_Dataset,
        imgpaths=[
            "tests/RSNA_test_data_jpg/folder",
            "tests/RSNA_test_data_jpg/tar.tar",
            "tests/RSNA_test_data_jpg/zip.zip",
            "tests/RSNA_test_data_jpg/zipped_1",
            "tests/RSNA_test_data_jpg/zipped_2",
            "tests/RSNA_test_data_jpg/tgz_1",
            "tests/RSNA_test_data_jpg/tgz_2"
        ],
        csvpath="tests/rsna_train.csv"
    )

def test_rsna_dcm_formats():
    _test_opening_formats(
        xrv.datasets.RSNA_Pneumonia_Dataset,
        imgpaths=[
            "tests/RSNA_test_data_dcm/folder",
            "tests/RSNA_test_data_dcm/tar.tar",
            "tests/RSNA_test_data_dcm/zip.zip",
            "tests/RSNA_test_data_dcm/zipped_1",
            "tests/RSNA_test_data_dcm/zipped_2",
            "tests/RSNA_test_data_dcm/tgz_1",
            "tests/RSNA_test_data_dcm/tgz_2"
        ],
        csvpath="tests/rsna_train.csv",
        extension=".dcm"
    )

def test_chex_formats():
    _test_opening_formats(
        xrv.datasets.CheX_Dataset,
        imgpaths=[
            "tests/CheXpert_test_data/folder",
            "tests/CheXpert_test_data/tar.tar",
            "tests/CheXpert_test_data/zip.zip",
            "tests/CheXpert_test_data/zipped_1",
            "tests/CheXpert_test_data/zipped_2",
            "tests/CheXpert_test_data/tgz_1",
            "tests/CheXpert_test_data/tgz_2"
        ],
        csvpath="tests/test_chexpert_data.csv"
    )

def test_COVID_dataset():
    _test_opening_formats(
        xrv.datasets.COVID19_Dataset,
        imgpaths=[
            "tests/COVID_test_data/folder",
            "tests/COVID_test_data/tar.tar",
            "tests/COVID_test_data/zip.zip",
            "tests/COVID_test_data/zipped_1",
            "tests/COVID_test_data/zipped_2",
            "tests/COVID_test_data/tgz_1",
            "tests/COVID_test_data/tgz_2"
        ],
        csvpath="tests/test_covid_data.csv"
    )

def test_openi_dataset():
    _test_opening_formats(
        xrv.datasets.Openi_Dataset,
        imgpaths=[
            "tests/Openi_test_data/folder",
            "tests/Openi_test_data/tar.tar",
            "tests/Openi_test_data/zip.zip",
            "tests/Openi_test_data/zipped_1",
            "tests/Openi_test_data/zipped_2",
            "tests/Openi_test_data/tgz_1",
            "tests/Openi_test_data/tgz_2"
        ],
        dicomcsv_path="tests/openi.csv"
    )

def test_mimic_formats():
    _test_opening_formats(
        xrv.datasets.MIMIC_Dataset,
        imgpaths=[
            "tests/gen_mimic/images-224/files",
            "tests/gen_mimic/images-224.tar",
            "tests/gen_mimic/images-224.zip",
            "tests/gen_mimic/images-224-zips_1",
            "tests/gen_mimic/images-224-zips_2",
            "tests/gen_mimic/images-224-tgzs_1",
            "tests/gen_mimic/images-224-tgzs_2"
        ],
        csvpath="tests/gen_mimic/mimic-cxr-2.0.0-negbio.csv",
        metacsvpath="tests/gen_mimic/mimic-cxr-2.0.0-metadata.csv"
    )

