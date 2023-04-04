import sys, os
sys.path.insert(0, "../torchxrayvision/")
import torchxrayvision as xrv
import pytest
from pydicom.data import get_testdata_file

file_path = os.path.abspath(os.path.dirname(__file__))
test_dcm_img_file = os.path.join(file_path, "1.2.276.0.7230010.3.1.4.8323329.6904.1517875201.850819.dcm")

# files used from the pydicom library
test_dcm_img_lut_file =  get_testdata_file("MR-SIEMENS-DICOM-WithOverlays.dcm")
test_dcm_img_monochr2_file = get_testdata_file("CT_small.dcm")
test_dcm_rgb_img = get_testdata_file("SC_rgb_jpeg.dcm")

# monochr1 created by @a-parida12
test_dcm_img_monochr1_file = os.path.join(file_path, "Fake_MONOHR1.dcm")

@pytest.mark.parametrize(
    "path, lut_config, monochrome_config, expected_pixel_value, warn",[
    # file -> bit depth: 8, no lut, monochrome2
    (test_dcm_img_file, False, False, -959.749, False),
    
    # file -> bit depth: 12, has lut, monochrome2
    # test -> application of lut for viewing
    (test_dcm_img_lut_file, True, False, -1024, False), 
    (test_dcm_img_lut_file, False, False, -1022, False), 
    
    # file -> bit depth: 16, has no lut, monochrome2
    # check monochrome2 is not modified when fix_monochrome=True
    (test_dcm_img_monochr2_file, False, True, -1007, False), 
    (test_dcm_img_monochr2_file, False, False, -1007, False),
    
    # file -> bit depth: 16, has no lut, monochrome1
    # check monochrome1 is modified when fix_monochrome=True
    # check for raise warning when modification happens
    (test_dcm_img_monochr1_file, False, True, 959.75, True),
    (test_dcm_img_monochr1_file, False, False, -959.75, False),
        ]
        
)
def test_dicomreader_end2end(path, lut_config, monochrome_config, expected_pixel_value, warn):
    out = xrv.utils.read_xray_dcm(path=path, voi_lut=lut_config, fix_monochrome=monochrome_config)
    obtained_pixel_value = float(out[60][2])
    assert obtained_pixel_value == pytest.approx(expected_pixel_value,0.001)
    assert out.max() <= 1024
    assert out.min() >= -1024
    if warn:
        with pytest.warns():
            xrv.utils.read_xray_dcm(path=path, voi_lut=lut_config, fix_monochrome=monochrome_config)

def test_dicomreader_photometric_interpretation():
    # PhotometricInterpretation MONOCHROME raises no error
    xrv.utils.read_xray_dcm(path=test_dcm_img_file)
   
    # raises an error when PhotometricInterpretation is RGB
    with pytest.raises(NotImplementedError):
        xrv.utils.read_xray_dcm(path=test_dcm_rgb_img)

    