from io import BytesIO
import numpy as np
from PIL import Image

import os
import cv2
import gc
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

from .clip import *

model = None
valid_df = pd.read_csv('application/components/prediction/valid_df.csv', index_col=0)

def load_model():
    model = CLIPModel()
    model.load_state_dict(torch.load('application/components/prediction/best.pt', map_location=torch.device('cpu')))
    return model

def predict(image: Image.Image):

    global model
    if model is None:
        model = load_model()
    model.eval()

    image = np.asarray(image)
    t = get_transforms()
    image = t(image=image)['image']
    image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)

    valid_image_query_embeddings = []

    with torch.no_grad():
        image_query_features = model.image_encoder(image.to(CFG.device))
        image_query_embeddings = model.image_projection(image_query_features)
        valid_image_query_embeddings.append(image_query_embeddings)

    image_embeddings, text_embeddings = get_image_text_embeddings()

    image_from_query_embeddings = torch.cat(valid_image_query_embeddings)
    image_from_query_embeddings_n = F.normalize(image_from_query_embeddings, p=2, dim=-1)

    ### Finding most similar image
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    dot_similarity = image_from_query_embeddings_n @ image_embeddings_n.T
    value, index_image = torch.topk(dot_similarity.squeeze(0), k=6)

    ### Finding most similar text
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = image_from_query_embeddings_n @ text_embeddings_n.T
    value, index_text = torch.topk(dot_similarity.squeeze(0), k=2)

    return valid_df.at[index_image[5].item(), 'image'], valid_df.at[index_text[1].item(), 'caption']


def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image
