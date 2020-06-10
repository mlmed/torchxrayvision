## torchxrayvision scripts


### process_image.py

This CLI will make a prediction using a pretrained model.

```
$ python process_image.py ../tests/16747_3_1.jpg 
{'preds': {'Atelectasis': 0.32797316,
           'Cardiomegaly': 0.36455172,
           'Consolidation': 0.42933327,
           'Edema': 0.02414272,
           'Effusion': 0.27805623,
           'Emphysema': 0.5011832,
           'Enlarged Cardiomediastinum': 0.2721874,
           'Fibrosis': 0.51887786,
           'Fracture': 0.51916164,
           'Hernia': 0.00993878,
           'Infiltration': 0.5316924,
           'Lung Lesion': 0.011150725,
           'Lung Opacity': 0.59073937,
           'Mass': 0.6392845,
           'Nodule': 0.68982005,
           'Pleural_Thickening': 0.24489857,
           'Pneumonia': 0.1856989,
           'Pneumothorax': 0.28849697}}
```

To extract the features extracted by the model run with `-feats` (1024 dimensional feats vector is shortened to fit)
```
$ python process_image.py ../tests/16747_3_1.jpg -feats
{'feats': [0.052227624, 0.014457048, 0.0, ...shortened...  0.1864362, 0.0, 0.9219263], 'preds': {'Atelectasis': 0.32797316, 
'Consolidation': 0.42933327, 'Infiltration': 0.5316924, 'Pneumothorax': 0.28849697, 'Edema': 0.02414272, 
'Emphysema': 0.5011832, 'Fibrosis': 0.51887786, 'Effusion': 0.27805623, 'Pneumonia': 0.1856989, 'Pleural_Thickening': 
0.24489857, 'Cardiomegaly': 0.36455172, 'Nodule': 0.68982005, 'Mass': 0.6392845, 'Hernia': 0.00993878, 'Lung Lesion': 
0.011150725, 'Fracture': 0.51916164, 'Lung Opacity': 0.59073937, 'Enlarged Cardiomediastinum': 0.2721874}}

```

The script will default to the `all` pre-trained model. To change this specify the `-weights` argument.
```
$ python process_image.py ../tests/16747_3_1.jpg -weights kaggle
{'preds': {'Atelectasis': nan,
           'Cardiomegaly': nan,
           'Consolidation': nan,
           'Edema': nan,
           'Effusion': nan,
           'Emphysema': nan,
           'Enlarged Cardiomediastinum': nan,
           'Fibrosis': nan,
           'Fracture': nan,
           'Hernia': nan,
           'Infiltration': nan,
           'Lung Lesion': nan,
           'Lung Opacity': 0.50366426,
           'Mass': nan,
           'Nodule': nan,
           'Pleural_Thickening': nan,
           'Pneumonia': 0.5038823,
           'Pneumothorax': nan}}
```
