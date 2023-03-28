import sys, os
sys.path.insert(0, "../torchxrayvision/")
import torchxrayvision as xrv
import pytest
from pydicom.data import get_testdata_file


file_path = os.path.abspath(os.path.dirname(__file__))
test_dcm_img_file = os.path.join(file_path, "1.2.276.0.7230010.3.1.4.8323329.6904.1517875201.850819.dcm")

@pytest.mark.parametrize(
        "path, lut_config, monochrome_config, expected_pixel_value",[
    (test_dcm_img_file, False, False, -1024.0),
    (test_dcm_img_file, True, False, -1024.0),
        ]
        
)
def test_pydicom_end2end(path, lut_config, monochrome_config, expected_pixel_value):

    out = xrv.utils.read_xray_dcm(path=path, voi_lut=lut_config, fix_monochrome=monochrome_config)
    assert out[0][0] ==  expected_pixel_value
    import pdb; pdb.set_trace()
