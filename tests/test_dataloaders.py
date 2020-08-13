import pytest
import pickle
import torchxrayvision as xrv
import os
from pathlib import Path

dataset_classes = [xrv.datasets.NIH_Dataset,
                   xrv.datasets.PC_Dataset,
                   xrv.datasets.NIH_Google_Dataset,
                   xrv.datasets.Openi_Dataset]

def test_dataloader_basic():
    for dataset_class in dataset_classes:
        dataset_class(imgpath=".")

def test_dataloader_merging():
    
    datasets = []
    for dataset_class in dataset_classes:
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
    
def test_mimic_tar():
    print(os.getcwd())
    #Load tarred and untarred datasets
    mimic_test_dir = Path("tests/gen_mimic")
    metacsvpath = mimic_test_dir/"mimic-cxr-2.0.0-metadata.csv"
    csvpath = mimic_test_dir/"mimic-cxr-2.0.0-negbio.csv"
    tarred = xrv.datasets.MIMIC_Dataset(
        imgpath=mimic_test_dir/"images-224.tar",
        csvpath=csvpath,
        metacsvpath=metacsvpath,
    )
    extracted = xrv.datasets.MIMIC_Dataset(
        imgpath=mimic_test_dir/"images-224"/"files",
        csvpath=csvpath,
        metacsvpath=metacsvpath
    )
    #Assert items are the same
    for tarred_item, extracted_item in zip(tarred, extracted):
        assert pickle.dumps(tarred_item) == pickle.dumps(extracted_item)

def all_equal(items):
    if len(items) == 1:
        return True
    return all([item == items[0] for item in items[1:]])

def _test_opening_formats(dataset_class, imgpaths, **kwargs):
    sources = [dataset_class(imgpath=path, **kwargs) for path in imgpaths]
    for one_item_from_each in zip(*sources):
        assert all_equal([pickle.dumps(item) for item in one_item_from_each])

def test_mimic_formats():
    _test_opening_formats(
        xrv.datasets.MIMIC_Dataset,
        imgpaths=[
            "tests/gen_mimic/images-224/files",
            "tests/gen_mimic/images-224.tar",
            "tests/gen_mimic/images-224.zip"
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
            "tests/NIH_test_data/zip.zip"
        ],
        csvpath="tests/nih.csv"
    )

def test_pc_formats():
    _test_opening_formats(
        xrv.datasets.PC_Dataset,
        imgpaths=[
            "tests/PC_test_data/folder",
            "tests/PC_test_data/tar.tar",
            "tests/PC_test_data/zip.zip"
        ],
        csvpath="tests/pc.csv"
    )
