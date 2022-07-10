#streamlit app version of process_image.py script 
#try with https://th.bing.com/th/id/OIP.CpxeJw58HJyWy8wa4G-ahwHaHZ?pid=ImgDet&rs=1 for example
#pip install streamlit
import streamlit as st
import torch
import torchvision, torchvision.transforms
import torchxrayvision as xrv
import skimage, skimage.io
import argparse  
st.set_page_config(page_title="process image using pretrained model")
inpF = st.file_uploader("Choose a file")
if inpF is not None:
  st.image(inpF.getvalue(), caption='uploaded image')
  img = skimage.io.imread(inpF)
  img = xrv.datasets.normalize(img, 255)
  parser = argparse.ArgumentParser()
  parser.add_argument('-weights', type=str,default="densenet121-res224-all")
  parser.add_argument('-feats', default=False, help='', action='store_true')
  parser.add_argument('-cuda', default=False, help='', action='store_true')
  parser.add_argument('-resize', default=False, help='', action='store_true')
  cfg = parser.parse_args()
  st.write(cfg)
  # Check that images are 2D arrays
  if len(img.shape) > 2:
      img = img[:, :, 0]
  if len(img.shape) < 2:
      st.write("error, dimension lower than 2 for image")
  # Add color channel
  img = img[None, :, :]
  # the models will resize the input to the correct size so this is optional.
  if cfg.resize:
      transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                  xrv.datasets.XRayResizer(224)])
  else:
      transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])
  img = transform(img)
  model = xrv.models.get_model(cfg.weights)
  output = {}
  with torch.no_grad():
      img = torch.from_numpy(img).unsqueeze(0)
      if cfg.cuda:
          img = img.cuda()
          model = model.cuda()
          
      if cfg.feats:
          feats = model.features(img)
          feats = F.relu(feats, inplace=True)
          feats = F.adaptive_avg_pool2d(feats, (1, 1))
          output["feats"] = list(feats.cpu().detach().numpy().reshape(-1))
      preds = model(img).cpu()
      output["preds"] = dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))
      st.write(output)
