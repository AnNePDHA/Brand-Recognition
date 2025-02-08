import requests
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import requests
import matplotlib.patches as patches
import faiss
import abc
import torch.nn as nn
from sklearn.metrics import pairwise_distances
from torch import Tensor
from typing import Union, List, Tuple, Callable, Optional, Any
import json
from sklearn.random_projection import SparseRandomProjection
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import glob
import pickle
import numpy as np
import cv2
from transformers import ViTImageProcessor, ViTModel, AutoProcessor, OwlViTForObjectDetection, OwlViTProcessor, VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from matplotlib.patches import Rectangle
import os
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torchvision import transforms
from sentence_transformers import SentenceTransformer, util


embeddings_dir = "weight/"


# Load class names map
with open(os.path.join(embeddings_dir, 'class_names_animation_map.pkl'), 'rb') as f:
    class_name_list = pickle.load(f)

# Add new to read image_paths
sample_emb_list = np.load(os.path.join(embeddings_dir, 'feature_embeddings.npy'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load CLIP model
feature_extractor = SentenceTransformer('clip-ViT-L-14')

feature_extractor.eval()
feature_extractor.to(device)

# Load owl vit model
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

model.to(device)
model.eval()


# build the index, d=size of vectors

d = sample_emb_list.shape[1]
index = faiss.IndexFlatL2(d) # with cpu
# index = faiss.GpuIndexFlatL2(d) # with gpu # []
# add vectors to the index -> []
index.add(sample_emb_list)


def anime_face_detector(img_path):
    # for img_path in img_paths:
    image = Image.open(img_path).convert("RGB")
    res_boxes = []
    # Process the image with the object detection model
    texts = [["a photo of animated characters faces"]]
    inputs = processor(text=texts, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract bounding box predictions
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
    i = 0
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    boxes = boxes.clamp(min=0.0).int()

    for box, conf in zip(boxes, scores):
        if conf > 0.13:
            res_boxes.append(box)
            
    return res_boxes

def anime_face_recogition(img_path, boxes):
    topk = 10
    test_img = Image.open(img_path).convert('RGB')
    predicted_clss_lst = []

    for box in boxes:
        test_emb_list = []
        default_label = "Other"
        max_conf = 0
        box = box.cpu().numpy()
        x1, y1, x2, y2 = box
        cropped_img = np.array(test_img, dtype=np.uint8)[y1:y2, x1:x2]

        with torch.no_grad():
            test_emb = feature_extractor.encode(Image.fromarray(cropped_img).convert('L'))[None]

        test_emb_list.append(test_emb)
        # test_img_path_list.append(img_path)
        test_emb_list = np.concatenate(test_emb_list, axis=0, dtype=np.float32)
        test_emb = test_emb_list[0][None]

        D_list, idx_list_for_image = index.search(test_emb, topk)

        cosine_sim_list = []

        for idx in idx_list_for_image[0]:
            sample_emb = sample_emb_list[idx][None]

            cosine_sim = F.cosine_similarity(
                    torch.tensor(test_emb, dtype=torch.float32),
                    torch.tensor(sample_emb, dtype=torch.float32)
            ).item()

            if cosine_sim > max_conf:
                max_conf = cosine_sim
                default_label = class_name_list[idx]
            # print(class_name_list[idx], cosine_sim)
            # cosine_sim_list.append(cosine_sim)
        # max_conf = max(cosine_sim_list)

        if max_conf >= 0.9:
            predicted_clss_lst.append((box, default_label, max_conf))
        elif (max_conf < 0.9) and (max_conf > 0.87):
            predicted_clss_lst.append((box, 'Maybe ' + default_label, max_conf))

        # print("==================================")
        # predicted_clss_lst.append((default_label, max_conf))
    
    return predicted_clss_lst

