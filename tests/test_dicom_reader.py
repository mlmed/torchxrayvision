import sys, os
sys.path.insert(0, "../torchxrayvision/")
import torchxrayvision as xrv
import pytest
from pydicom.data import get_testdata_file

file_path = os.path.abspath(os.path.dirname(__file__))
test_dcm_img_file = os.path.join(file_path, "1.2.276.0.7230010.3.1.4.8323329.6904.1517875201.850819.dcm")
test_dcm_img_lut_file =  get_testdata_file("MR-SIEMENS-DICOM-WithOverlays.dcm")
test_dcm_img_monochr2_file = get_testdata_file("CT_small.dcm")
# created by @a-parida12
test_dcm_img_monochr1_file = os.path.join(file_path, "Fake_MONOHR1.dcm")

@pytest.mark.parametrize(
        "path, lut_config, monochrome_config, expected_pixel_value, warn",[
    (test_dcm_img_file, False, False, 8, False),
    (test_dcm_img_lut_file, True, False, 0, False),
    (test_dcm_img_lut_file, False, False, 2, False),
    (test_dcm_img_monochr2_file, False, True, 538, False),
    (test_dcm_img_monochr2_file, False, False, 538, False),
    (test_dcm_img_monochr1_file, False, True, 246, True),
    (test_dcm_img_monochr1_file, False, False, 8, False),
        ]
        
)
def test_pydicom_end2end(path, lut_config, monochrome_config, expected_pixel_value, warn):
    out = xrv.utils.read_xray_dcm(path=path, voi_lut=lut_config, fix_monochrome=monochrome_config)
    assert (out[60][2] - expected_pixel_value) == 0
    if warn :
        with pytest.warns():
            xrv.utils.read_xray_dcm(path=path, voi_lut=lut_config, fix_monochrome=monochrome_config)


    