import pytest
import sys, os
import torchxrayvision as xrv
import numpy as np
sys.path.insert(0,"../torchxrayvision")


@pytest.fixture(scope="session", autouse=True)
def resource(request):
    print("setup")
    os.system('git clone --depth=1 https://github.com/ieee8023/covid-chestxray-dataset /tmp/covid-chestxray-dataset')
    
    def teardown():
        print("teardown")
        os.system("rm -rf /tmp/covid-chestxray-dataset")
    request.addfinalizer(teardown)
    

def test_covid_dataloader_basic():
    d_covid19 = xrv.datasets.COVID19_Dataset(imgpath="/tmp/covid-chestxray-dataset/images/",
                                        csvpath="/tmp/covid-chestxray-dataset/metadata.csv",
                                        views=['PA', 'AP','AP Supine'])
    
    print(d_covid19)


def test_covid_dataloader_get():
    
    d_covid19 = xrv.datasets.COVID19_Dataset(imgpath="/tmp/covid-chestxray-dataset/images/",
                                        csvpath="/tmp/covid-chestxray-dataset/metadata.csv",
                                        views=['PA', 'AP','AP Supine'])
    
    # pick 5 random
    for i in np.random.choice(range(len(d_covid19)),5):
        print(i)
        d_covid19[i]
