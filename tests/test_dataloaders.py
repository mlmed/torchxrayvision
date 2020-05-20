import pytest
import torchxrayvision as xrv
from skimage.io import imread, imsave

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
    
    
def test_dataloader_merging():
    
    for filename in ["16747_3_1.jpg", "covid-19-pneumonia-58-prior.jpg"]
        img = imread(filename)
        img = xrv.datasets.normalize(img, 255)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]

        # Add color channel
        img = img[None, :, :]    

        resize_ski = xrv.datasets.XRayResizer(100, engine="skimage")
        resize_cv2 = xrv.datasets.XRayResizer(100, engine="cv2")

        assert(np.allclose(resize_ski(img),resize_cv2(img)))
