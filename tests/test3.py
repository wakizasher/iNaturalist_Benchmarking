import torch
from PIL import Image
import open_clip
from timm.data.tf_preprocessing import preprocess_image
import numpy
import open_clip
import pandas as pd


model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-vit-b-16-inat-only')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-vit-b-16-inat-only')
