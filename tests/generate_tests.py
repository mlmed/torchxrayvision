import torchxrayvision as xrv

with open("test_indices") as handle:
    indices = [int(line) for line in handle]

create_tests_for = [
    (xrv.datasets.CheX_Dataset,
     {"imgpath":"/network/tmp1/paul.morrison/network/CheXpert-v1.0-small.zip",
      "csvpath":}
    ),
    (xrv.datasets.MIMIC_Dataset,
     {}
    ),
    (xrv.datasets.NIH_Dataset,
    ),
    (xrv.datasets.NIH_Google_Dataset,
    ),
    (xrv.datasets.NLMTB_Dataset,
    ),
    (xrv.datasets.Openi_Dataset,
    ),
    (xrv.datasets.PC_Dataset,
    ),
    (xrv.datasets.RSNA_Pneumonia_Dataset,
    ),
    (xrv.datasets.COVID19_Dataset,
    )
]

for dataset in create_tests_for:
    print(dataset.__name__)
    print(dataset().csv[:10])
