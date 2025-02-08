import cv2
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pickle
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import OwlViTForObjectDetection, OwlViTProcessor
from torchvision.ops import box_iou
from sentence_transformers import SentenceTransformer

dirty_fix = False

if dirty_fix:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

root_dir = ""
embeddings_dir = "weight/"
TOPK_SEARCH = 5

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

# Logo Comparation

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

feature_extractor = SentenceTransformer('clip-ViT-L-14')

feature_extractor.eval()
feature_extractor.to(device)

# feature_embeddings = np.load(os.path.join(embeddings_dir, "all-8000.npy"))

# Load feature embeddings
# with open(os.path.join(embeddings_dir, 'feature_embeddings_logo_list.pkl'), 'rb') as f:
#     feature_embeddings = pickle.load(f)

# Load class names map

feature_embeddings_list = np.load(os.path.join(embeddings_dir, "feature_embeddings_logo_list.pkl.npy"))

feature_embeddings = np.concatenate(feature_embeddings_list, axis=0)

# Load feature embeddings
# with open(os.path.join(embeddings_dir, 'feature_embeddings_logo_list.pkl'), 'rb') as f:
#     feature_embeddings = pickle.load(f)

# Load class names map
with open(os.path.join(embeddings_dir, 'class_embs_logo_name.pkl'), 'rb') as f:
    sample_names = pickle.load(f)

# Load class names map
with open(os.path.join(embeddings_dir, 'resnet_logo_preprocess.pkl'), 'rb') as f:
    real_img_path = pickle.load(f)

# build the index, d=size of vectors
d = feature_embeddings.shape[1]
# index = faiss.IndexFlatL2(d) # with cpu
# index = faiss.GpuIndexFlatL2(d) # with gpu

index = faiss.IndexFlatIP(d) # with cpu
# index = faiss.GpuIndexFlatIP(d) # with gpu

# normalize input vectors
faiss.normalize_L2(feature_embeddings)

# add vectors to the index
index.add(feature_embeddings)

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

def nms(boxes, scores, threshold):
    """
    Perform Non-Maximum Suppression to filter out overlapping boxes.

    Args:
    - boxes (Tensor): Bounding boxes of shape (N, 4) where N is the number of boxes.
    - scores (Tensor): Confidence scores associated with each bounding box.
    - threshold (float): Threshold value for IoU (Intersection over Union) to consider boxes as overlapping.

    Returns:
    - selected_indices (Tensor): Indices of the selected boxes after NMS.
    """
    selected_indices = []

    # Sort boxes by their scores
    _, indices = scores.sort(descending=True)

    while indices.numel() > 0:
        # Select box with highest confidence (first box in sorted list)
        selected_indices.append(indices[0].item())
        if indices.numel() == 1:
            break

        # Calculate IoU between the selected box and other boxes
        ious = box_iou(boxes[indices[0].unsqueeze(0)], boxes[indices[1:]])

        # Filter out boxes with IoU greater than threshold
        mask = ious.squeeze(0) <= threshold
        indices = indices[1:][mask]

    return torch.tensor(selected_indices, dtype=torch.long)

def logo_detection(img_path):
    image = Image.open(img_path).convert('RGBA')
    image = np.array(image)
    image = convert_RGBA_to_RGB(image)

    texts = [["a photo of car brand logos"], ["a photo of app platform logos"], ["a photo of sport brand logos"], ["a photo of soft drink logos"], ["a photo of restaurant logos"]]
    
    # texts = [["a photo of animated characters faces"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.Tensor([image.shape[:-1:]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.03)
    i = 0
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    nms_threshold =0.4
    new_box_index =  nms(boxes, scores, nms_threshold)
    boxes = boxes[new_box_index]
    
    return boxes

def logo_recognition(img_path, boxes):
    image = Image.open(img_path).convert('RGBA')
    image = np.array(image)
    image = convert_RGBA_to_RGB(image)
    predicted_logo_results = []
    
    for box in boxes:
        default_label = "Other"

        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = [max(coord, 0) for coord in [x1, y1, x2, y2]]
        cropped_img = np.array(image, dtype=np.uint8)[y1:y2, x1:x2]
        hed = convert_crop_into_hed(cropped_img)

        # hed_lists.append(hed)
        with torch.no_grad():
            test_emb2 = feature_extractor.encode(Image.fromarray(hed))[None]
    
        cosine_similarity_list, idx_list_for_image = index.search(test_emb2, 1)
        retrieved_index = idx_list_for_image[0][0]
        cosine_sim = F.cosine_similarity(
                        torch.tensor(feature_embeddings[retrieved_index], dtype=torch.float32),
                        torch.tensor(test_emb2, dtype=torch.float32)
            ).item()
    
        # print(cosine_sim)
        # Draw rectangle if cosine similarity is greater than 0.9
        if cosine_sim > 0.86:


            output1 = convert_resnes(hed)
    
            output2 = real_img_path[retrieved_index]
            # output2 = convert_resnes(hed_sample)
    
            # output2 = convert_resnes(hed_list[retrieved_index])
    
            resnet_cosine_sim = F.cosine_similarity(
                    torch.tensor(output1, dtype=torch.float32),
                    torch.tensor(output2, dtype=torch.float32)
            ).item()
            if resnet_cosine_sim >= 0.86:
                default_label = sample_names[retrieved_index]

                predicted_logo_results.append((box, default_label))
    
            

    return predicted_logo_results