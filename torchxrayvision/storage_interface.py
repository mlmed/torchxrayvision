from PIL import Image
import os
from hashlib import blake2b
import pickle
import zipfile
import tarfile
import multiprocessing
import pydicom
from pathlib import Path
import numpy as np
from io import BytesIO

Image.init() #loads image extensions

"""
You can read agnostically from folders, zipfiles, and tarfiles with this submodule.
You can retrieve each file using just the last n elements of its path (you pick n).

You can create an interface using create_interface(imgpath, n). The path to the
archive or folder is imgpath, and the path length you will use for retrieving files
is n.

Then, you can retrieve each file with the .get_image() method of the returned object.

Example:

interface = create_interface("/path/to/images.tar", n = 3)
interface.get_image("element1/element2/element3.jpg") #note n = 3

The "interface" object will belong to one of four classes:
    TarInterface    - for tarfiles
    ZipInterface    - for zipfiles
    FolderInterface - for folders containing images
    ArchiveFolder   - for folders containing multiple tarfiles/zipfiles.

"""


def last_n_in_filepath(filepath, n):
    """
    Return the last n pieces of a path (takes a string, not a Path object).
    For example:
    last_n_in_filepath("a/b/c",2) -> "b/c"
    """
    if n < 1:
        return ""
    start_part, end_part = os.path.split(filepath)
    for i in range(n - 1):
        start_part, middle_part = os.path.split(start_part)
        end_part = os.path.join(middle_part, end_part)
    return end_part

def get_filename_mapping_path(imgpath, path_length):
    """
    Create a hash of (imgpath, last_modification, path_length_for_mapping_key)
    and use it to return the filepath for a cached index.
    """
    imgpath = str(imgpath) #cannot be Path object
    imgpath = os.path.abspath(imgpath)
    timestamp = os.path.getmtime(imgpath)
    length = path_length
    key = (imgpath, timestamp, length)

    #Construct filename from hash of imgpath, timestamp, and length
    cache_filename = str(blake2b(pickle.dumps(key)).hexdigest()) + ".pkl"

    file_mapping_cache_folder = os.path.expanduser(os.path.join(
        "~", ".torchxrayvision", "filename-mapping-cache"
    ))

    filename_mapping_path = os.path.join(file_mapping_cache_folder, cache_filename)

    return filename_mapping_path

def load_filename_mapping(imgpath, path_length):
    "If a cached filename mapping exists, return it. Otherwise, return None"

    filename_mapping_path = get_filename_mapping_path(imgpath, path_length)

    if os.path.exists(filename_mapping_path):
        print("Loading indexed file paths from cache")
        with open(filename_mapping_path, "rb") as handle:
            filename_mapping = pickle.load(handle)
    else:
        filename_mapping = None

    return filename_mapping

def save_filename_mapping(imgpath, path_length, filename_mapping):
    "Load the dataset's index from the cache if available, else create a new one."

    filename_mapping_path = get_filename_mapping_path(imgpath, path_length)

    try:
        #Pickle filename_mapping.
        os.makedirs(os.path.dirname(filename_mapping_path), exist_ok=True)
        with open(filename_mapping_path, "wb") as handle:
            pickle.dump(filename_mapping, handle)
        return True

    except:
        return False

def convert_to_image(filename, bytes):
    "Convert an image byte array to a numpy array. If the filename ends with .dcm, use pydicom."
    if str(filename).endswith(".dcm"):
        return pydicom.filereader.dcmread(BytesIO(bytes), force=True).pixel_array
    else:
        return np.array(Image.open(BytesIO(bytes)))

class StorageInterface(object):
    pass

class TarInterface(StorageInterface):
    "This class supports extracting files from a tar archive based on a partial path."
    @classmethod
    def matches(cls, filename):
        "Return whether the given path is a tar archive."
        return not os.path.isdir(filename) and tarfile.is_tarfile(filename)
    def __init__(self, imgpath, path_length):
        "Store the archive path, and the length of the partial paths within the archive"
        self.path_length = path_length
        self.imgpath = imgpath

        #Load archive and filename mapping
        compressed = None
        self.filename_mapping = load_filename_mapping(imgpath, path_length)
        #If the filename mapping could not be loaded, create it and save it
        if self.filename_mapping is None:
             compressed, self.filename_mapping = self.index(imgpath)
             save_filename_mapping(imgpath, path_length, self.filename_mapping)
        #If the compressed file has still not been loaded, load it.
        if compressed is None:
             compressed = tarfile.open(imgpath)
        self.all_compressed = {multiprocessing.current_process().name:compressed}

    def get_image(self, imgid):
        "Return the image object for the partial path provided."
        archive_path = self.filename_mapping[imgid]
        if not multiprocessing.current_process().name in self.all_compressed:
            #print("Opening tar file on thread:",pid)
            # check and reset number of open files if too many
            if len(self.all_compressed.keys()) > 64:
                self.all_compressed = {}
            self.all_compressed[multiprocessing.current_process().name] = tarfile.open(self.imgpath)
        bytes = self.all_compressed[multiprocessing.current_process().name].extractfile(archive_path).read()
        return convert_to_image(archive_path, bytes)
    def index(self, imgpath):
        "Create a dictionary mapping imgpath -> path within archive"
        print("Indexing file paths (one-time). The next load will be faster")
        compressed = tarfile.open(imgpath)
        tar_infos = compressed.getmembers()
        filename_mapping = {}
        for tar_info in tar_infos:
            if tar_info.type != "DIRTYPE":
                tar_path = tar_info.name
                imgid = last_n_in_filepath(tar_path, self.path_length)
            filename_mapping[imgid] = tar_path
        return compressed, filename_mapping
    def close(self):
        "Close all open tarfiles."
        for compressed in self.all_compressed.values():
            compressed.close()

class ZipInterface(StorageInterface):
    "This class supports extracting files from a zip archive based on a partial path."
    @classmethod
    def matches(cls, filename):
        "Return whether the given path is a zip archive."
        return not os.path.isdir(filename) and zipfile.is_zipfile(filename)
    def __init__(self, imgpath, path_length):
        "Store the archive path, and the length of the partial paths within the archive"
        self.path_length = path_length
        self.imgpath = imgpath

        #Load archive and filename mapping
        compressed = None
        self.filename_mapping = load_filename_mapping(imgpath, path_length)
        #If the filename mapping could not be loaded, create it and save it
        if self.filename_mapping is None:
             compressed, self.filename_mapping = self.index(imgpath)
             save_filename_mapping(imgpath, path_length, self.filename_mapping)
        #If the compressed file has still not been loaded, load it.
        if compressed is None:
             compressed = zipfile.ZipFile(imgpath)
        self.all_compressed = {multiprocessing.current_process().name:compressed}

    def get_image(self, imgid):
        "Return the image object for the partial path provided."
        archive_path = self.filename_mapping[imgid]
        if not multiprocessing.current_process().name in self.all_compressed:
            #print("Opening zip file on thread:",multiprocessing.current_process())
            # check and reset number of open files if too many
            if len(self.all_compressed.keys()) > 64:
                self.all_compressed = {}
            self.all_compressed[multiprocessing.current_process().name] = zipfile.ZipFile(self.imgpath)
        bytes = self.all_compressed[multiprocessing.current_process().name].open(archive_path).read()
        return convert_to_image(archive_path, bytes)
    def index(self, imgpath):
        "Create a dictionary mapping imgpath -> path within archive"
        print("Indexing file paths (one-time). The next load will be faster")
        compressed = zipfile.ZipFile(imgpath)
        zip_infos = compressed.infolist()
        filename_mapping = {}
        for zip_info in zip_infos:
            if not zip_info.is_dir():
                zip_path = zip_info.filename
                imgid = last_n_in_filepath(zip_path, self.path_length)
                filename_mapping[imgid] = zip_path
        return compressed, filename_mapping
    def close(self):
        "Close all open zipfiles."
        for compressed in self.all_compressed.values():
            compressed.close()

class FolderInterface(StorageInterface):
    "This class supports drawing files from a folder based on a partial path."

    @classmethod
    def matches(cls, filename):
        "Return whether the given path is a zip archive."
        return os.path.isdir(filename)

    def __init__(self, imgpath, path_length):
        "Store the archive path, and the length of the partial paths within the archive"
        self.path_length = path_length

        self.filename_mapping = load_filename_mapping(imgpath, path_length)
        #If the filename mapping could not be loaded, create it and save it
        if self.filename_mapping is None:
             _, self.filename_mapping = self.index(imgpath)
             save_filename_mapping(imgpath, path_length, self.filename_mapping)

    def get_image(self, imgid):
        "Return the image object for the partial path provided."
        archive_path = self.filename_mapping[imgid]
        with open(archive_path,"rb") as handle:
            image = convert_to_image(archive_path, handle.read())
        return image
    def index(self, imgpath):
        "Create a dictionary mapping imgpath -> path within archive"
        print("Indexing file paths (one-time). The next load will be faster")
        filename_mapping = {}
        for path in Path(imgpath).rglob("*"):
            if not os.path.isdir(path):
                imgid = last_n_in_filepath(path, self.path_length)
                filename_mapping[imgid] = path
        return imgpath, filename_mapping
    def close(self):
        pass

def is_image(filename):
    "Return whether the given filename has an image extension."
    _, extension = os.path.splitext(filename)
    return extension in Image.EXTENSION

archive_interfaces = [ZipInterface, TarInterface]

def is_archive(filename):
    "Return whether the given filename is a tarfile or zipfile."
    return any(interface.matches(filename) for interface in archive_interfaces)


class ArchiveFolder(StorageInterface):
    "This class supports extracting files from multiple tar or zip archives under the same root directory."

    @classmethod
    def matches(cls, filename):
        for item in Path(filename).rglob("*"):
            if is_image(item):
                return False
            if is_archive(item):
                return True
        return False

    def __init__(self, imgpath, path_length):
        "Store the archive path, and the length of the partial paths within the archive"
        self.path_length = path_length
        self.archives = None
        self.filename_mapping = load_filename_mapping(imgpath, path_length)
        #If the filename mapping could not be loaded, create it and save it
        if self.filename_mapping is None:
             self.archives, self.filename_mapping = self.index(imgpath)
             save_filename_mapping(imgpath, path_length, self.filename_mapping)
        #If the compressed file has still not been loaded, load it.
        if self.archives is None:
             self.archives = self.get_archive(imgpath)

    def get_image(self, imgid):
        "Return the image object for the partial path provided."
        path_to_archive = self.filename_mapping[imgid]
        return self.archives[path_to_archive].get_image(imgid)

    def index(self, filename):
        """
        Create a dictionary mapping imgid -> containing sub-archive.
        The archives are identified by their filenames. The sub-archive
        will then be queried itself.

        This is different from the index method of ZipInterface and
        TarInterface, where the dictionary values are the actual file
        paths.
        """
        archives = self.get_archive(filename)
        filename_mapping = {}
        for path_to_archive, archive in archives.items():
            for path_in_csv, path_in_archive in archive.filename_mapping.items():
                filename_mapping[path_in_csv] = path_to_archive
        return archives, filename_mapping

    def get_archive(self, filename):
        archives = {}
        for path_to_archive in Path(filename).rglob("*"):
            if is_archive(path_to_archive):
                archive = create_interface(path_to_archive, self.path_length)
                archives[path_to_archive] = archive
        return archives

    def close(self):
        "Recursively close all open archives."
        for archive_path, archive in self.archives.items():
            archive.close()

interfaces = [ArchiveFolder, FolderInterface, TarInterface, ZipInterface]

def create_interface(filename, path_length, interfaces=interfaces):
    "Choose the right interface type for the given path, and return an initialized interface."
    for interface in interfaces:
        if interface.matches(filename):
            return interface(filename, path_length)
