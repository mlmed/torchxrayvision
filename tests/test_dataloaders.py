import os
import shutil
import sys

import numpy as np
import pytest
import torchvision

import torchxrayvision as xrv

sys.path.insert(0, "../torchxrayvision/")


file_path = os.path.abspath(os.path.dirname(__file__))

dataset_classes = [xrv.datasets.NIH_Dataset,
                   xrv.datasets.PC_Dataset,
                   xrv.datasets.NIH_Google_Dataset,
                   xrv.datasets.Openi_Dataset,
                   xrv.datasets.CheX_Dataset,
                   xrv.datasets.SIIM_Pneumothorax_Dataset,
                   xrv.datasets.VinBrain_Dataset]

dataset_classes_pydicom = [xrv.datasets.SIIM_Pneumothorax_Dataset,
                           xrv.datasets.VinBrain_Dataset]

test_data_path = "/tmp/testdata"
test_png_img_file = os.path.join(file_path, "00000001_000.png")
test_jpg_img_file = os.path.join(file_path, "16747_3_1.jpg")
test_dcm_img_file = os.path.join(file_path, "1.2.276.0.7230010.3.1.4.8323329.6904.1517875201.850819.dcm")


@pytest.fixture
def is_pydicom_installed():
    try:
        import pydicom
        return True
    except:
        return False

def get_clazz_imgpath(clazz):
    return os.path.join(test_data_path, clazz.__name__)


def create_test_img(test_img_file, clazz, filename):
    imgpath = get_clazz_imgpath(clazz)
    os.makedirs(os.path.join(imgpath, os.path.dirname(filename)))
    shutil.copyfile(test_img_file, os.path.join(imgpath, filename))


@pytest.fixture(scope="session", autouse=True)
def create_test_images(request):

    if os.path.exists(test_data_path):
        shutil.rmtree(test_data_path)

    # for nih dataset
    create_test_img(test_png_img_file, xrv.datasets.NIH_Dataset, "00000001_000.png")
    create_test_img(test_png_img_file, xrv.datasets.PC_Dataset, "100014625199913409730274754282179594842_0jycky.png")
    create_test_img(test_png_img_file, xrv.datasets.NIH_Google_Dataset, "00000211_006.png")
    create_test_img(test_png_img_file, xrv.datasets.Openi_Dataset, "CXR10_IM-0002-1001.png")
    create_test_img(test_jpg_img_file, xrv.datasets.CheX_Dataset, "train/patient00004/study1/view1_frontal.jpg")
    create_test_img(test_dcm_img_file, xrv.datasets.SIIM_Pneumothorax_Dataset, "1.2.276.0.7230010.3.1.2.8323329.6904.1517875201.850818/1.2.276.0.7230010.3.1.3.8323329.6904.1517875201.850817/1.2.276.0.7230010.3.1.4.8323329.6904.1517875201.850819.dcm")
    create_test_img(test_dcm_img_file, xrv.datasets.VinBrain_Dataset, "000434271f63a053c4128a0ba6352c7f.dicom")


def test_dataloader_basic(create_test_images, is_pydicom_installed):

    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])

    for dataset_class in dataset_classes:
        if is_pydicom_installed or (dataset_class not in dataset_classes_pydicom):
            dataset = dataset_class(imgpath=get_clazz_imgpath(dataset_class), transform=transform)

            sample = dataset[0]

            assert("img" in sample)
            assert("lab" in sample)
            assert("idx" in sample)


def test_dataloader_merging(is_pydicom_installed):

    datasets = []
    for dataset_class in dataset_classes:
        if is_pydicom_installed or (dataset_class not in dataset_classes_pydicom):
            dataset = dataset_class(imgpath=".")
            datasets.append(dataset)

    for dataset in datasets:
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, dataset)

    xrv.datasets.MergeDataset(datasets)

    # also test alias
    xrv.datasets.Merge_Dataset(datasets)


def test_dataloader_merging_dups():

    datasets = []
    for dataset_class in dataset_classes:
        dataset = dataset_class(imgpath=".")
        datasets.append(dataset)

    for dataset in datasets:
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, dataset)

    for dataset in datasets:
        xrv.datasets.Merge_Dataset([dataset, dataset])

    # now merge merge datasets
    for dataset in datasets:
        dd = xrv.datasets.Merge_Dataset([dataset, dataset])
        xrv.datasets.Merge_Dataset([dd, dd])


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
    
    
    
def test_dataloader_relabelling(create_test_images):
    
    d_nih = xrv.datasets.NIH_Dataset(imgpath=get_clazz_imgpath(xrv.datasets.NIH_Dataset))
    xrv.datasets.relabel_dataset(['Mass'], d_nih)
    
    assert d_nih[0]['lab'] == d_nih.labels[0]
    
    
    
def test_errors_when_doing_things_that_should_not_work():
    
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                xrv.datasets.XRayResizer(224)])
    
    data_aug = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.5, 1.5)),
        torchvision.transforms.ToTensor()
    ])
    
    datasets = []
    for dataset_class in dataset_classes:
        dataset = dataset_class(imgpath=".", transform=transform, data_aug=data_aug)
        datasets.append(dataset)

    for dataset in datasets:
        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, dataset)

    merged_dataset = xrv.datasets.MergeDataset(datasets)
    
    with pytest.raises(NotImplementedError) as excinfo:
        merged_dataset.transform = None
        
    with pytest.raises(NotImplementedError) as excinfo:
        merged_dataset.data_aug = None
        
    with pytest.raises(NotImplementedError) as excinfo:
        merged_dataset.labels = None
        
    with pytest.raises(NotImplementedError) as excinfo:
        merged_dataset.pathologies = None
    
    subset_dataset = xrv.datasets.SubsetDataset(datasets[0], [0,1,2])
    
    with pytest.raises(NotImplementedError) as excinfo:
        subset_dataset.transform = None
        
    with pytest.raises(NotImplementedError) as excinfo:
        subset_dataset.data_aug = None
        
    with pytest.raises(NotImplementedError) as excinfo:
        merged_dataset.labels = None
        
    with pytest.raises(NotImplementedError) as excinfo:
        merged_dataset.pathologies = None


def test_relabel_dataset_missing_pathology_becomes_nan():
    """Pathologies requested but absent in the dataset must produce all-NaN columns."""
    d = xrv.datasets.NIH_Dataset(imgpath=".")
    n_samples = len(d)

    # Pick a pathology that NIH_Dataset definitely does NOT have
    absent_pathology = "Enlarged Cardiomediastinum"
    assert absent_pathology not in d.pathologies, \
        f"{absent_pathology} unexpectedly present in NIH_Dataset"

    xrv.datasets.relabel_dataset([absent_pathology], d, silent=True)

    assert d.pathologies == [absent_pathology]
    assert d.labels.shape == (n_samples, 1)
    assert np.all(np.isnan(d.labels[:, 0])), \
        "Missing pathology column should be all NaN after relabeling"


def test_relabel_dataset_present_pathology_preserved():
    """Pathologies that exist in the dataset must retain their original label values."""
    d = xrv.datasets.NIH_Dataset(imgpath=".")

    present_pathology = "Atelectasis"
    assert present_pathology in d.pathologies

    original_col_idx = list(d.pathologies).index(present_pathology)
    original_col = d.labels[:, original_col_idx].copy()

    xrv.datasets.relabel_dataset([present_pathology], d, silent=True)

    assert d.pathologies == [present_pathology]
    np.testing.assert_array_equal(d.labels[:, 0], original_col)


def test_relabel_dataset_dropped_pathologies_removed():
    """Pathologies not in the requested list must be absent after relabeling."""
    d = xrv.datasets.NIH_Dataset(imgpath=".")
    original_pathologies = list(d.pathologies)

    keep = [original_pathologies[0]]
    xrv.datasets.relabel_dataset(keep, d, silent=True)

    assert list(d.pathologies) == keep
    assert d.labels.shape[1] == 1


def test_merge_dataset_lab_alignment():
    """MergeDataset.labels must be the vertical concatenation of the sub-dataset
    labels in order, so that merged.labels[i] equals the correct sub-dataset row."""
    d1 = xrv.datasets.NIH_Dataset(imgpath=".")
    d2 = xrv.datasets.NIH_Dataset(imgpath=".")

    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d1, silent=True)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d2, silent=True)

    merged = xrv.datasets.MergeDataset([d1, d2])

    # First len(d1) rows come from d1
    np.testing.assert_array_equal(merged.labels[:len(d1)], d1.labels)
    # Next len(d2) rows come from d2
    np.testing.assert_array_equal(merged.labels[len(d1):], d2.labels)

    # Spot-check: the which_dataset index and offset arithmetic must match labels
    for idx in [0, len(d1) - 1, len(d1), len(merged) - 1]:
        ds_idx = merged.which_dataset[idx]
        local_idx = idx - int(merged.offset[idx])
        expected_lab = merged.datasets[ds_idx].labels[local_idx]
        np.testing.assert_array_equal(
            merged.labels[idx],
            expected_lab,
            err_msg=f"merged.labels[{idx}] doesn't match sub-dataset label"
        )


def test_merge_dataset_source_field():
    """MergeDataset.which_dataset must record the correct sub-dataset index for every row."""
    d1 = xrv.datasets.NIH_Dataset(imgpath=".")
    d2 = xrv.datasets.NIH_Dataset(imgpath=".")

    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d1, silent=True)
    xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, d2, silent=True)

    merged = xrv.datasets.MergeDataset([d1, d2])

    assert merged.which_dataset[0] == 0
    assert merged.which_dataset[len(d1) - 1] == 0
    assert merged.which_dataset[len(d1)] == 1
    assert merged.which_dataset[len(merged) - 1] == 1
