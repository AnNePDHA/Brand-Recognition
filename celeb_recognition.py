import face_recognition
import numpy as np
import pickle
import faiss

import os
import cv2
from model.deepface import DeepFace
from collections import Counter


kwargs = {
    'img_size': (640, 640),
    'strides': [8, 16, 32],
    'conf_thresh': 0.5,
    'nms_thresh': 0.45,
    'use_gpu': True
}

db_path = "test/data_200" # thu muc chua anh tao database
model_name = "facenet" # chon model facenet cho face recog
distance_metric = "l2" # chon phuong phap tinh toan khac biet -> chon l2 
detector_backend = "rtcdet" # chon model rtcdet cho face detect
topk = 3 # chon top k de lay ra 3 anh tot nhat gan giong voi giong voi face co trong anh verify
result_folder = "results" # noi de luu tru ket qua
threshold = 0.74 # threshold cho face recog

def celebrities_recognition(img_path):
    predicted_class_lst = []
    resp_objs = DeepFace.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        distance_metric=distance_metric,
        detector_backend=detector_backend,
        align=False,
        threshold=threshold,
        **kwargs
    )
    # verify_img = cv2.imread(img_path)
    # result_img_path = os.path.join(result_folder, os.path.basename(img_path))

    for resp_obj in resp_objs:
        if not resp_obj.empty:
            # title = f"{model_name}_{detector_backend}_Threshold_{distance_metric}: {resp_obj.iloc[0]['threshold']:.2f}"

            folder_counts = Counter(os.path.basename(os.path.dirname(
                img_path)) for img_path in resp_obj['identity'])
            
            # filtered_items = [(item, count) for item, count in folder_counts.items() if count > 2]

            # Sort the filtered items by count (descending)
            # sorted_filtered_items = sorted(filtered_items, key=lambda x: x[1], reverse=True)
            
            topk_folders = [(folder, count) for folder, count in folder_counts.most_common(topk) if count > 7]
            
            # print(folder_counts)
            # print(len(folder_counts))
            # print(topk_folders)
        
            top_scores = resp_obj.iloc[0]['distance']
    
            if top_scores < 0.5:
                
                label = os.path.basename(os.path.dirname(resp_obj.iloc[0]['identity']))
                
                source_bbox = (resp_obj.iloc[0]['source_x'], resp_obj.iloc[0]['source_y'],
                                resp_obj.iloc[0]['source_w'], resp_obj.iloc[0]['source_h'])
                
                predicted_class_lst.append((source_bbox, label))
                

            elif topk_folders != []:
                num_count = topk_folders[0][1]
                source_bbox = (resp_obj.iloc[0]['source_x'], resp_obj.iloc[0]['source_y'],
                                resp_obj.iloc[0]['source_w'], resp_obj.iloc[0]['source_h'])
                label = topk_folders[0][0]
                if len(topk_folders) == 1:
                    # label = "Maybe " + topk_folders[0][0]
                    predicted_class_lst.append((source_bbox, label))


                
            # if num_count > 1:
            #     label = topk_folders[0][0]
            #     if num_count < 7:
            #         label = "Maybe " + topk_folders[0][0]

            #     predicted_class_lst.append((source_bbox, label))
                # draw_bbox(verify_img, source_bbox, color=(
                #     255, 0, 0), folder_label= label)

    return predicted_class_lst