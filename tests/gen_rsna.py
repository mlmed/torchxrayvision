from argparse import ArgumentParser
from random_data import write_random_images
import pandas as pd

rsna_imgid_column = "patientId"

extension = ".jpg"

def gen_rsna(test_csv, train_csv, test_data_folder, dimensions):
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)
    write_random_images(
        pd.concat([
            train_data[rsna_imgid_column].map(lambda path: os.path.join(train_folder, path)),
            test_data[rsna_imgid_column].map(lambda path: os.path.join(train_folder, path))
        ]) + extension,
        test_data_folder/"folder",
        test_data_folder/"tar.tar",
        test_data_folder/"zip.zip",
        dimensions
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("test_data_folder")
    parser.add_argument("x")
    parser.add_argument("y")
    args = parser.parse_args()
    gen_rsna(
        args.test,
        args.train,
        args.test_data_folder,
        (int(args.x), int(args.y))
    )
