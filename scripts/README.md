## torchxrayvision scripts


### process_image.py

This CLI will make a prediction using a pretrained model.

```
$ python process_image.py ../tests/16747_3_1.jpg -resize
{'preds': {'Atelectasis': 0.32797325,
           'Cardiomegaly': 0.36455178,
           'Consolidation': 0.42933366,
           'Edema': 0.02414272,
           'Effusion': 0.27805617,
           'Emphysema': 0.5011832,
           'Enlarged Cardiomediastinum': 0.27218732,
           'Fibrosis': 0.51887786,
           'Fracture': 0.5191617,
           'Hernia': 0.00993878,
           'Infiltration': 0.5316924,
           'Lung Lesion': 0.0111507205,
           'Lung Opacity': 0.5907394,
           'Mass': 0.63928443,
           'Nodule': 0.68982005,
           'Pleural_Thickening': 0.24489847,
           'Pneumonia': 0.18569912,
           'Pneumothorax': 0.2884971}}
```

To extract the features extracted by the model run with `-feats` (1024 dimensional feats vector is shortened to fit)
```
$ python process_image.py ../tests/16747_3_1.jpg -resize -feats
{'feats': [0.052227624, 0.014457048, 0.0, ...shortened...  0.1864362, 0.0, 0.9219263], 'preds': {'Atelectasis': 0.32797316, 
'Consolidation': 0.42933327, 'Infiltration': 0.5316924, 'Pneumothorax': 0.28849697, 'Edema': 0.02414272, 
'Emphysema': 0.5011832, 'Fibrosis': 0.51887786, 'Effusion': 0.27805623, 'Pneumonia': 0.1856989, 'Pleural_Thickening': 
0.24489857, 'Cardiomegaly': 0.36455172, 'Nodule': 0.68982005, 'Mass': 0.6392845, 'Hernia': 0.00993878, 'Lung Lesion': 
0.011150725, 'Fracture': 0.51916164, 'Lung Opacity': 0.59073937, 'Enlarged Cardiomediastinum': 0.2721874}}

```

The script will default to the `all` pre-trained model. To change this specify the `-weights` argument.
```
$ python process_image.py ../tests/16747_3_1.jpg -resize -weights densenet121-res224-rsna
{'preds': {'Atelectasis': 0.5,
           'Cardiomegaly': 0.5,
           'Consolidation': 0.5,
           'Edema': 0.5,
           'Effusion': 0.5,
           'Emphysema': 0.5,
           'Enlarged Cardiomediastinum': 0.5,
           'Fibrosis': 0.5,
           'Fracture': 0.5,
           'Hernia': 0.5,
           'Infiltration': 0.5,
           'Lung Lesion': 0.5,
           'Lung Opacity': 0.50366426,
           'Mass': 0.5,
           'Nodule': 0.5,
           'Pleural_Thickening': 0.5,
           'Pneumonia': 0.5038823,
           'Pneumothorax': 0.5}}
```




More reference runs:

```bash
$ python process_image.py ../tests/00000001_000.png 
Warning: Input size (512x512) is not the native resolution (224x224) for this model. A resize will be performed but this could impact performance.
{'preds': {'Atelectasis': 0.5064204,
           'Cardiomegaly': 0.6068723,
           'Consolidation': 0.32152757,
           'Edema': 0.20760165,
           'Effusion': 0.40498933,
           'Emphysema': 0.5032768,
           'Enlarged Cardiomediastinum': 0.46466783,
           'Fibrosis': 0.5410498,
           'Fracture': 0.29533696,
           'Hernia': 0.013291723,
           'Infiltration': 0.5222817,
           'Lung Lesion': 0.22291252,
           'Lung Opacity': 0.43178767,
           'Mass': 0.42115793,
           'Nodule': 0.50882816,
           'Pleural_Thickening': 0.510605,
           'Pneumonia': 0.13716412,
           'Pneumothorax': 0.33823627}}

$ python process_image.py ../tests/00000001_000.png -weights resnet50-res512-all
{'preds': {'Atelectasis': 0.056815326,
           'Cardiomegaly': 0.44829446,
           'Consolidation': 0.009982704,
           'Edema': 0.0050414177,
           'Effusion': 0.1477132,
           'Emphysema': 0.0041440246,
           'Enlarged Cardiomediastinum': 0.5,
           'Fibrosis': 0.017102875,
           'Fracture': 0.035871908,
           'Hernia': 0.002605671,
           'Infiltration': 0.12528789,
           'Lung Lesion': 0.5,
           'Lung Opacity': 0.008149157,
           'Mass': 0.0124504855,
           'Nodule': 0.028969117,
           'Pleural_Thickening': 0.032002825,
           'Pneumonia': 0.006612198,
           'Pneumothorax': 0.002571446}}


$ python process_image.py ../tests/16747_3_1.jpg 
Warning: Input size (885x885) is not the native resolution (224x224) for this model. A resize will be performed but this could impact performance.
{'preds': {'Atelectasis': 0.33015457,
           'Cardiomegaly': 0.375611,
           'Consolidation': 0.4268645,
           'Edema': 0.019578112,
           'Effusion': 0.28957728,
           'Emphysema': 0.5007649,
           'Enlarged Cardiomediastinum': 0.2713981,
           'Fibrosis': 0.5198858,
           'Fracture': 0.52066934,
           'Hernia': 0.0097252205,
           'Infiltration': 0.52939886,
           'Lung Lesion': 0.014771313,
           'Lung Opacity': 0.6033015,
           'Mass': 0.64704525,
           'Nodule': 0.69085634,
           'Pleural_Thickening': 0.25272927,
           'Pneumonia': 0.08875252,
           'Pneumothorax': 0.26505184}}

$ python process_image.py ../tests/16747_3_1.jpg -weights resnet50-res512-all
Warning: Input size (885x885) is not the native resolution (512x512) for this model. A resize will be performed but this could impact performance.
{'preds': {'Atelectasis': 0.02676412,
           'Cardiomegaly': 0.011656165,
           'Consolidation': 0.028988084,
           'Edema': 0.004503417,
           'Effusion': 0.025871767,
           'Emphysema': 0.011733518,
           'Enlarged Cardiomediastinum': 0.5,
           'Fibrosis': 0.0036442582,
           'Fracture': 0.05079817,
           'Hernia': 0.0010036419,
           'Infiltration': 0.20743765,
           'Lung Lesion': 0.5,
           'Lung Opacity': 0.821504,
           'Mass': 0.053269155,
           'Nodule': 0.15826094,
           'Pleural_Thickening': 0.023301573,
           'Pneumonia': 0.05978322,
           'Pneumothorax': 0.015683603}}

```


