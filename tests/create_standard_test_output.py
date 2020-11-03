import torchxrayvision as xrv
import os

def create_standard_test_output(dataset, name, columns = None, n=10):
    minimal_dataset = dataset.csv[:n]
    if columns is not None:
        minimal_dataset = minimal_dataset[columns]
    minimal_dataset.to_csv(name, index=False)
