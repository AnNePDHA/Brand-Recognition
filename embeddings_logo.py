import cv2
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import pickle
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from torchvision.ops import box_iou
from sentence_transformers import SentenceTransformer
import traceback
from sys import platform

embeddings_dir = "weight/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

# load WRN-50-2:
# res_net_model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
# or WRN-101-2
res_net_model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True)
res_net_model.eval()

# Lodo Detection
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

net = cv2.dnn.readNetFromCaffe("model/logodetection/deploy.prototxt", "model/logodetection/hed_pretrained_bsds.caffemodel")

feature_extractor = SentenceTransformer('clip-ViT-L-14')

feature_extractor.eval()
feature_extractor.to(device)


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Convert crop_img into hed
def convert_crop_into_hed(image):
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(500, 500),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)

    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (image.shape[1], image.shape[0]))
    hed = (255 * hed).astype("uint8")

    # convert into 3 channels
    hed_rgb = np.repeat(hed[:, :, np.newaxis], 3, axis=2)
    return hed_rgb

# Handle 4 channel images
def convert_RGBA_to_RGB(input_image):
    # Takes an RGBA image as input
    input_image_normed = input_image / 255  # shape (nx, ny, 4), dtype float
    alpha = input_image_normed[..., -1:]  # shape (nx, ny, 1) for broadcasting
    input_image_normed_rgb = input_image_normed[..., :-1]  # shape (nx, ny, 3)
    #bg_normed = np.zeros_like(input_image_normed_rgb)  # shape (nx, ny, 3) <-- black background
    bg_normed = np.ones_like(input_image_normed_rgb)  # shape (nx, ny, 3) <-- white background
    composite_normed = (1 - alpha) * bg_normed + alpha * input_image_normed_rgb
    composite = (composite_normed * 255).round().astype(np.uint8)
    return composite



def convert_resnes(hed):
    input_tensor = preprocess(Image.fromarray(hed))
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        res_net_model.to('cuda')

    with torch.no_grad():
        output = res_net_model(input_batch)

    return output

def get_template_with_color(temp_path, color=(168, 240, 234)):
    temp_img = Image.open(temp_path)
    bands_list = temp_img.getbands() # [R, G, B], [R, G, B, A] or [L] or [P]
    ndim = len(bands_list)
    if ndim == 1 and bands_list[0] == 'P':
        temp_img = temp_img.convert('RGBA')
        ndim = 4
    if ndim == 4:
        temp_img_with_garment_color = Image.new("RGB", temp_img.size, color)
        temp_img_with_garment_color.paste(temp_img, mask=temp_img.split()[3])
    else:
        temp_img_with_garment_color = temp_img.convert('RGB')
    return temp_img_with_garment_color


# @title get all image paths
sample_paths = glob.glob(os.path.join('Logos/Logos/', '*'))

all_image_paths = []

for folder_path in sample_paths:
    # Use glob to get all image files within each folder
    folder_image_paths = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))

    # Extend the list of all_image_paths with the paths of images in the current folder
    all_image_paths.extend(folder_image_paths)

bright_paths = [path for path in all_image_paths if "_bright." in path]


resnet_preprocess = []
feature_embeddings_list = []
class_embs_name = []
for img_path in bright_paths:
# for img_path in test_paths_sample:
    try:
        img = get_template_with_color(img_path)
        img_array = np.array(img)
        hed = convert_crop_into_hed(img_array)
        output = convert_resnes(hed)
        resnet_preprocess.append(output)
        with torch.no_grad():
            embeddings = feature_extractor.encode(Image.fromarray(hed))[None]
        feature_embeddings_list.append(embeddings)
        if platform == 'win32':
            class_embs_name.append(img_path.split("\\")[-2])

        else:
            class_embs_name.append(img_path.split(r'/')[-2])

        print(img_path)
    except Exception as e:
        traceback.print_exc()
        print(f"Error processing image {img_path}: {e}")


with open(os.path.join(embeddings_dir,"class_embs_logo_name.pkl"), 'wb') as file:
    pickle.dump(class_embs_name, file)

with open(os.path.join(embeddings_dir,"resnet_logo_preprocess.pkl"), 'wb') as file:
    pickle.dump(resnet_preprocess, file)

# with open(os.path.join(embeddings_dir,"feature_embeddings_logo_list.pkl"), 'wb') as file:
#     pickle.dump(feature_embeddings_list, file)
# Save the numpy array to the .npy file
np.save(os.path.join(embeddings_dir,"feature_embeddings_logo_list.pkl"), feature_embeddings_list)