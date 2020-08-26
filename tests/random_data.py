import tarfile
import datetime
import numpy as np
import zipfile
import os
import shutil
import random
from PIL import Image
import glob
import pdb
import pydicom
import io
from pydicom.dataset import FileMetaDataset, FileDataset
from pydicom.encaps import encapsulate
import pydicom
import tempfile

import io
from PIL import Image, ImageDraw
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid, JPEGExtended
from pydicom._storage_sopclass_uids import SecondaryCaptureImageStorage
from pydicom import dcmread
from pydicom.encaps import encapsulate
import numpy as np

def np_to_dcm(image, filename):
    image = np.array(image)
    WIDTH = image.shape[1]
    HEIGHT = image.shape[2]
    ds = Dataset()
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.fix_meta_info()
    ds.Modality = "OT"
    ds.SamplesPerPixel = 3
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PlanarConfiguration = 1
    ds.PhotometricInterpretation = "RGB"
    ds.Rows = HEIGHT
    ds.Columns = WIDTH
    ds.PixelData = encapsulate([image.tobytes()])
    ds["PixelData"].is_undefined_length = True
    ds.PhotometricInterpretation = "YBR_FULL_422"
    ds.file_meta.TransferSyntaxUID = JPEGExtended
    ds.save_as(filename, write_like_original=False)

#def np_to_dcm(arr, filename):
#    #Create object corresponding to file
#    meta = FileMetaDataset()
#    meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
#    meta.MediaStorageSOPInstanceUID = "1.2.3"
#    meta.ImplementationClassUID = "1.2.3.4"
#    dataset = FileDataset(filename, {}, file_meta = meta, preamble=b"\0" * 128)
#    dataset.PatientName = "Test^Firstname"
#    dataset.PatientID = "123456"
#    #ds.is_little_endian = True
#    dataset.is_little_endian = True
#    dataset.is_implicit_VR = True
#    dataset.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRBigEndian
#    # Set creation date/time
#    dt = datetime.datetime.now()
#    dataset.ContentDate = dt.strftime('%Y%m%d')
#    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
#    dataset.ContentTime = timeStr

#    dataset.Rows, dataset.Columns = arr.size

#    #ds.SOPClassUID = pydicom._storage_sopclass_uids.MRImageStorage
#    dataset.PatientName = "Test^Firstname"
#    dataset.PatientID = "123456"

#    #ds.Modality = "CT"
#    #ds.SeriesInstanceUID = pydicom.uid.generate_uid()
#    #ds.StudyInstanceUID = pydicom.uid.generate_uid()
#    #ds.FrameOfReferenceUID = pydicom.uid.generate_uid()

#    dataset.BitsStored = 16
#    dataset.BitsAllocated = 16
#    dataset.SamplesPerPixel = 1
#    dataset.HighBit = 15
#    #ds.SliceLocation = DCM_SliceLocation
#    #ds.SpacingBetweenSlices = 1
#    #ds.SliceThickness = 4
#    #ds.ScanLength = length

#    dataset.ImagesInAcquisition = "1"

#    dataset.InstanceNumber = 1

#    #ds.ImagePositionPatient = r"-159\-174"+ "\\-" + str(DCM_SliceLocation*4)  #default of 6, sometimes 1
#    #ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
#    #ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

#    dataset.RescaleIntercept = "0"
#    dataset.RescaleSlope = "1"
#    dataset.PixelSpacing = r"0.683594\0.683594"# r"1\1"
#    dataset.PhotometricInterpretation = "MONOCHROME2"
#    dataset.PixelRepresentation = 1

#    #Store image as bytes
#    bytes_img = io.BytesIO()
#    arr.save(bytes_img, format="PNG")
#    #Add byte image to file
#    dataset.PixelData = encapsulate(bytes_img.read())
#    pdb.set_trace()
#    #dataset.pixel_data = np.array(arr)
#    #Write file
#    dataset.save_as(filename)

def save_as_dicom(arr, filename):
    file = FileDataset()
    file.binary_data = arr
    img_bytes = io.BytesIO()
    Image.fromarray(arr).save(img_bytes, format="PNG")
    file.PixelData = img_bytes
    file.save_as(filename)

def generate_random_image(dimensions):
    dimensions = tuple(dimensions)
    if len(dimensions) == 2:
        dimensions = dimensions + (3,)
    return Image.fromarray((np.random.random(dimensions)*255).astype("uint8"))

class FolderOfArchive:
    folder_format = "folder{}"
    depth = 1
    def __init__(self, root, depth, archive_size=3):
        self.root = root
        self.depth = depth
        self.archive_size = archive_size
        self.current_archive = -1
        self.archive_position = archive_size - 1
        self.archives = []
    def get_path_from_root(self, n):
        return os.path.join(*(
            [self.folder_format.format(n)] * self.depth + \
            [self.archive_format.format(n)]
        ))
    def get_current_archive(self):
        self.archive_position += 1
        if self.archive_position == self.archive_size:
            new_path = os.path.join(
                self.root,
                self.get_path_from_root(self.current_archive)
            )
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            self.archives.append(self.get_new_archive(new_path))
            self.archive_position = 0
            self.current_archive += 1
        return self.archives[-1]
    def close(self):
        for archive in self.archives:
            archive.close()
    def write(self, content):
        curr = self.get_current_archive()
        self.add_to_archive(curr, content)

class FolderOfTar(FolderOfArchive):
    archive_format = "tar{}.tar"
    def add_to_archive(self, archive, content):
        archive.add(content)
    def get_new_archive(self, new_path):
        return tarfile.open(new_path, "w")

class FolderOfTarGz(FolderOfArchive):
    archive_format = "tar{}.tar.gz"
    def add_to_archive(self, archive, content):
        archive.add(content)
    def get_new_archive(self, new_path):
        return tarfile.open(new_path, "w:gz")

class FolderOfZip(FolderOfArchive):
    archive_format = "zip{}.zip"
    def add_to_archive(self, archive, content):
        archive.write(content)
    def get_new_archive(self, new_path):
        return zipfile.ZipFile(new_path, "w")

def write_random_images(paths, extracted, tarname, zipname, folder_of_zip_name, folder_of_tar_gz_name, dimensions, subfolder="."):
    folder_of_zip_d1_name = str(folder_of_zip_name) + "_1"
    folder_of_zip_d2_name = str(folder_of_zip_name) + "_2"
    folder_of_tar_gz_d1_name = str(folder_of_tar_gz_name) + "_1"
    folder_of_tar_gz_d2_name = str(folder_of_tar_gz_name) + "_2"
    for path in [extracted, tarname, zipname, folder_of_zip_d1_name, folder_of_zip_d2_name]:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else: #dir
                shutil.rmtree(path)
    for img_fname in paths:
        print(img_fname)
        img_path = extracted/subfolder/img_fname
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        random_image = generate_random_image(dimensions)
        if str(img_fname).endswith(".dcm"):
            np_to_dcm(random_image, img_path)
        else:
            random_image.save(img_path)
    tarred = tarfile.TarFile.open(tarname, "w")
    zipped = zipfile.ZipFile(zipname,"w")
    folder_of_zip_d1 = FolderOfZip(folder_of_zip_d1_name, 0)
    folder_of_zip_d2 = FolderOfZip(folder_of_zip_d2_name, 1)
    folder_of_tar_gz_d1 = FolderOfTarGz(folder_of_tar_gz_d1_name, 0)
    folder_of_tar_gz_d2 = FolderOfTarGz(folder_of_tar_gz_d2_name, 1)
    for file in extracted.rglob("*"):
        if not os.path.isdir(file):
            tarred.add(file)
            zipped.write(file)
            folder_of_zip_d1.write(file)
            folder_of_zip_d2.write(file)
            folder_of_tar_gz_d1.write(file)
            folder_of_tar_gz_d2.write(file)
    tarred.close()
    zipped.close()
    folder_of_zip_d1.close()
    folder_of_zip_d2.close()
    folder_of_tar_gz_d1.close()
    folder_of_tar_gz_d2.close()

def gen_hex(n):
    hex_chars = list("0123456789abcdef")
    return "".join(np.random.choice(hex_chars,n))

def gen_int(n):
    int_chars = list("0123456789abcdef")
    return "".join(np.random.choice(int_chars,n))

def random_pred():
    return random.choice(["1.0","-1.0","0.0",""])

def random_preds():
    return {
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
