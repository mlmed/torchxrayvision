python3 generate_test_data.py pc.csv ImageID 2 2 PC_test_data
python3 generate_test_data.py nih.csv "Image Index" 2 2 NIH_test_data
python3 generate_test_data.py openi.csv imageid 2 2 Openi_test_data .png
python3 generate_test_data.py shenzen.csv fname 2 2 Shenzen_test_data --subfolder CXR_png
python3 generate_test_data.py montgomery.csv fname 2 2 Montgomery_test_data --subfolder CXR_png
python3 generate_test_data.py rsna_train.csv patientId 2 2 RSNA_test_data_jpg .jpg --subfolder stage_2_train_images
python3 generate_test_data.py rsna_train.csv patientId 2 2 RSNA_test_data_dcm .dcm --subfolder stage_2_train_images
python3 generate_test_data.py test_chexpert_data.csv Path 2 2 CheXpert_test_data
python3 generate_test_data.py test_covid_data.csv filename 2 2 COVID_test_data --subfolder images
