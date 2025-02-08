import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import glob
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load CLIP model


feature_extractor = SentenceTransformer('clip-ViT-L-14')

feature_extractor.eval()
feature_extractor.to(device)


embeddings_dir = "weight"

# Load all sample for comparation
sample_emb_list = np.load(os.path.join(embeddings_dir, 'sample_emb_list_album_500.npy'))

with open(os.path.join(embeddings_dir, 'sample_paths_500.pkl'), 'rb') as f:
    sample_paths = pickle.load(f)

with open(os.path.join(embeddings_dir, 'image_arrays_500.pkl'), 'rb') as f:
    image_arrays = pickle.load(f)


lst_name_file = []
for img_path in sample_paths:
    file_name = Path(img_path).name
    lst_name_file.append(file_name)

d = sample_emb_list.shape[1]
index = faiss.IndexFlatL2(d) # with cpu
# index = faiss.GpuIndexFlatL2(d) # with gpu # []
# add vectors to the index -> []
index.add(sample_emb_list)

def image_matching_function(target_image_path, topk = 1):
     
     cosine_sim_threshold = 0.92
     # Read the image and convert it to grayscale
     test_img = Image.open(target_image_path).convert('L')
     # Convert the image to array
     test_img_array = np.array(test_img)

     test_emb_list = []
     # Perform encoding
     with torch.no_grad():
          test_emb = feature_extractor.encode(Image.fromarray(test_img_array))[None]

     # Append the encoded representation to the list
     test_emb_list.append(test_emb)
     # Now test_emb_tensor contains the encoded representations of all cropped images
     test_emb_list = np.concatenate(test_emb_list, axis=0, dtype=np.float32)
     test_emb = test_emb_list[0][None]
     
     ref_img_list = []
     cosine_sim_list = []

     D_list, idx_list_for_image = index.search(test_emb, topk)

     for idx in idx_list_for_image[0]:
          sample_emb = sample_emb_list[idx][None]
          ref_img = image_arrays[idx]

          cosine_sim = F.cosine_similarity(
                    torch.tensor(test_emb, dtype=torch.float32),
                    torch.tensor(sample_emb, dtype=torch.float32)
          ).item()

          cosine_sim_list.append(cosine_sim)
          ref_img_list.append(ref_img)

     max_value = max(cosine_sim_list)


     
     if max_value > cosine_sim_threshold:
          # print(lst_name_file[idx], max_value)
          return lst_name_file[idx]
     return []