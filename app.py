from flask import Flask, render_template, request, redirect
import tempfile
import numpy as np
import cv2
import base64
from azure_face_detection import azure_face_recognition
from PIL import Image
import os
import io
import face_recognition
import pickle
from werkzeug.utils import secure_filename
import shutil
from sys import platform

from text_recognition import analyze_text_in_image
from celeb_recognition import celebrities_recognition
from logo_recognition import logo_detection, logo_recognition
from image_matching import image_matching_function
from animation_recognition import anime_face_detector, anime_face_recogition
from fuzzywuzzy import fuzz

import pandas as pd

from datetime import datetime, timedelta
import json

os.umask(0)

# Read the CSV file into a DataFrame
df = pd.read_csv('list_text.csv')
# Access the data in the DataFrame
list_text = df['text']

app = Flask(__name__)

MAX_IMAGE_SIZE = 6 * 1024 * 1024

@app.route('/')
def index():
    return render_template('index.html')


def has_transparency(img):
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info) or (img.info.get("transparency", None) is not None):
        return True

    return False

 
@app.route('/upload', methods=['POST'])
def upload():
    # prev_umask = os.umask(0)

    current_time = secure_filename(str(datetime.now() + timedelta(hours=7)))

    if platform == 'win32':
        new_folder = f'./result/{current_time}'

    else:
        new_folder = f"/test/tsc-image/result/{current_time}"

    os.mkdir(new_folder)
    os.mkdir(os.path.join(new_folder, 'Input'))
    os.mkdir(os.path.join(new_folder, 'Output'))

    # Get images from front-end
    if 'files[]' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('files[]')

    if not files or all(file.filename == '' for file in files):
        return redirect(request.url)

    # Define reusults return to front-end
    result_images = []
    all_results = []
    text_results = []
    logo_bool_results = []
    filenames = []
    animation_results = []
    taylor_in_text = []
    image_matching_result_list = []

    counter = 1

    

    # Process images
    for file in files:
        # Get the current time

        filename_list = file.filename.rsplit('.', 1)
        original_extension = 'jpg' if filename_list[1].lower() == 'jpeg' else filename_list[1].lower()

        imencode_extension = '.' + original_extension

        temp_img = tempfile.NamedTemporaryFile(delete=False, dir='temp-location/', suffix=imencode_extension)
        temp_img.close()

        file.save(temp_img.name)

        input_filename = f'input{counter}{imencode_extension}'
        initial_raw_img = cv2.imread(temp_img.name)
        cv2.imwrite(os.path.join(new_folder, 'Input', input_filename), initial_raw_img)
        
        # #resize image
        # raw_image_size = os.stat(temp_img.name).st_size
        #
        # if raw_image_size > MAX_IMAGE_SIZE:
        #     # print('### Size overload, reducing...')
        #     quality = [95, 85, 75]
        #     chosen_quality = 75
        #
        #     # if has_transparency(img):
        #     #     print(f"### Image {file.filename} is transparent. ###")
        #
        #     # else:
        #     converted_img = Image.open(temp_img.name)
        #     # Convert to RGB
        #     converted_img = converted_img.convert('RGB')
        #
        #     for i in quality:
        #         bytes_arr = io.BytesIO()
        #         converted_img.save(bytes_arr, format='JPEG', quality=i)
        #
        #         if bytes_arr.__sizeof__() <= MAX_IMAGE_SIZE:
        #             chosen_quality = i
        #             break
        #
        #     converted_img.save(temp_img.name, format='JPEG', quality=chosen_quality)
        #     # print('Chosen quality:', chosen_quality)
        #     # print('Converted image size:', os.stat(converted_temp_img.name).st_size)
        #
        #     # chosen_file_name = converted_temp_img.name
        #     imencode_extension = '.jpg'

            # convert = True

        # # Face detection
        # list_faces = azure_face_recognition(temp_img.name)  # [top, left, width, height]

        # transparent_src_img = img.convert('RGBA')
        #
        # # Create a new white RGBA image with the same size as the original image
        # new_image = Image.new("RGBA", transparent_src_img.size, "Silver")
        #
        # # Paste the original image onto the new white image
        # new_image.paste(transparent_src_img, mask=transparent_src_img)
        #
        # # Convert the result to RGB mode and save it
        # converted_img = new_image.convert("RGB")
        #
        # format_extension = 'JPEG' if imencode_extension == '.jpg' or imencode_extension == '.jpeg' else 'PNG'
        # print('### Current format extension:', format_extension)
        #
        # if format_extension == 'JPEG':
        #     converted_img.save(temp_img.name, format=format_extension, quality=95)
        #
        # else:
        #     converted_img.save(temp_img.name, format=format_extension)

        # Detect text
        raw_image = cv2.imread(temp_img.name)
        _, img_encoded_original = cv2.imencode(imencode_extension, raw_image)

        print("*** Running text detection... ***")
        text_detected = analyze_text_in_image(img_encoded_original.tobytes())
        # text_detected = None

        # Append "text" result to result
        combined_content = ""
        max_char_per_line = 120

        if text_detected is not None:
            for block in text_detected.blocks:
                for line in block.lines:
                    line_content = line.text
                    combined_content += f"{line_content} "

            text_split = [combined_content[i: i + max_char_per_line] for i in range(0, len(combined_content), max_char_per_line)]
            combined_content = text_split

        # print("Text detected: ", text_detected)
        print("Combine Text: ",combined_content)
        text_results.append(combined_content)

        ## Detect keywords in text:
        full_text = ''.join(combined_content).lower()
        taylor_status = []

        for item in list_text:
            if len(item) <= len(full_text):
                score = fuzz.partial_ratio(item.lower(), full_text)

                if score > 95:
                    taylor_status.append(str(item))
                elif score > 80:
                    taylor_status.append("Maybe '" + str(item) + "'")

        if not taylor_status:
            taylor_status = 'Not detected yet.'

        taylor_in_text.append(str(taylor_status))
        print('*** Text detection done ***')

        # Face detection
        # list_faces = azure_face_recognition(temp_img.name)  # [top, left, width, height]
        # if convert:
        #     os.remove(converted_temp_img.name)


        # Detect celebrities
        # celebs_result_lst, face_locations = celebrities_recognition(temp_img.name, list_faces)
        print('*** Running celeb detection ***')
        final_celebs_result = celebrities_recognition(temp_img.name)
        celebs_names_lst = [res[1] for res in final_celebs_result]

        if not final_celebs_result:
            celebs_names_lst.append('Not detected yet.')

        # Append "celebrities" result to result
        all_results.append(celebs_names_lst)

        print('*** Celeb detection done ***')

        print('*** Running logo detection ***')
        # Detect logos
        logo_detect_results = logo_detection(temp_img.name)
        logo_recog_results = logo_recognition(temp_img.name, logo_detect_results)
        print('*** Logo detection done ***')

        ## Animation recognition
        print('*** Running anim detection ***')
        # Detect animation
        animation_boxes = anime_face_detector(temp_img.name)

        # Recognition animation
        animation_recog_results = anime_face_recogition(temp_img.name, animation_boxes)

        animation_result = [i[1] for i in animation_recog_results]

        final_animation_result = [i for i in animation_result if i != 'Other']
        if not final_animation_result:
            final_animation_result.append('Not detected yet.')

        animation_results.append(final_animation_result)
        print('*** Anim detection done ***')

        print('*** Running image matching... ***')
        ## Detect images matching in dataset for special case
        image_matching_result = []
        if celebs_names_lst == ['Not detected yet.'] and logo_recog_results == [] and str(taylor_status) == 'Not detected yet.' and final_animation_result == ['Not detected yet.']:
            # print("Cheking!")
            image_matching_result = image_matching_function(temp_img.name)
        
        image_matching_result_list.append(image_matching_result)
        print('*** Image matching done... ***')

        print('*** Running transparent image handler... ***')
        ### Transparent image changer ###
        img = Image.open(temp_img.name)
        transparent_src_img = img.convert('RGBA')

        # Create a new white RGBA image with the same size as the original image
        new_image = Image.new("RGBA", transparent_src_img.size, "Silver")

        # Paste the original image onto the new white image
        new_image.paste(transparent_src_img, mask=transparent_src_img)

        # Convert the result to RGB mode and save it
        converted_img = new_image.convert("RGB")

        format_extension = 'JPEG' if imencode_extension == '.jpg' or imencode_extension == '.jpeg' else 'PNG'
        print('### Current format extension:', format_extension)

        if format_extension == 'JPEG':
            converted_img.save(temp_img.name, format=format_extension, quality=95)

        else:
            converted_img.save(temp_img.name, format=format_extension)

        print('*** Transparent image handler done ***')
        raw_image = cv2.imread(temp_img.name)

        ## Draw bbox for famous celebrities
        for face_res in final_celebs_result:
            if face_res != 'Not detected yet.':
                x, y, w, h = face_res[0]
                clss_predicted = face_res[1]

                text_size = cv2.getTextSize(clss_predicted, 0, fontScale=1, thickness=1)[0]  # Convert label to string
                background = x + text_size[0], y - text_size[1] - 3

                cv2.rectangle(raw_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                cv2.rectangle(raw_image, (x, y), background, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.putText(raw_image, clss_predicted, (x, y - 2), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

        # Default result check logo "False"
        # logo_recog_bool = False
        logo_recog_bool = "Not detected yet."

        # Draw bbox for logos
        for logo in logo_recog_results:

            logo_bbox = logo[0]
            logo_label = logo[1]
            # conf_score = logo[2]
            # print(f"### LOGO DETECTED: {logo_label} | CONFIDENCE: {conf_score}")

            x1, y1, x2, y2 = map(int, logo_bbox)
            x1, y1, x2, y2 = [max(coord, 0) for coord in [x1, y1, x2, y2]]

            # Check having logo recog in image
            if logo_recog_bool != "Not detected yet.":
                logo_recog_bool += f', {logo_label}'

            else:
                logo_recog_bool = logo_label


            # x1, y1, x2, y2 = round(logo_bbox[0]), round(logo_bbox[1]), round(logo_bbox[2]), round(logo_bbox[3])


            text_size = cv2.getTextSize(str(logo_label), 0, fontScale=1, thickness=1)[0]  # Convert label to string
            background = x1 + text_size[0], y1 - text_size[1] - 3

            cv2.rectangle(raw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(raw_image, (x1, y1), background, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(raw_image, str(logo_label), (x1, y1 - 2), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
        
        # If having logo in image append True else append False in every image
        logo_bool_results.append(logo_recog_bool)


        # Draw bbox for animation
        for cartoon in animation_recog_results:

            cartoon_bbox = cartoon[0]
            cartoon_label = cartoon[1]
            # cartoon_conf = cartoon[2]

            # Check having logo recog in image


            x1, y1, x2, y2 = round(cartoon_bbox[0]), round(cartoon_bbox[1]), round(cartoon_bbox[2]), round(cartoon_bbox[3])

            # print(y1, x2, y2, x1)

            text_size = cv2.getTextSize(str(cartoon_label), 0, fontScale=1, thickness=1)[0]  # Convert label to string
            background = x1 + text_size[0], y1 - text_size[1] - 3

            cv2.rectangle(raw_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # cv2.rectangle(raw_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.rectangle(raw_image, (x1, y1), background, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(raw_image, str(cartoon_label), (x1, y1 - 2), 0, 1, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)
            # cv2.rectangle(raw_image, (x1, y1), background, (0, 255, 0), -1, cv2.LINE_AA)
            # cv2.putText(raw_image, str(logo_label), (x1, y1 - 2), 0, 1 , [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

        # print(image_matching_result)
        # print(image_matching_result_list)
        if len(image_matching_result) > 0:
            height, width = raw_image.shape[:2]
            cv2.rectangle(raw_image, (0, 0), (width - 1, height - 1), (0, 165, 255), 10)
        

        _, img_encoded_original = cv2.imencode(imencode_extension, raw_image)
        original_image_base64 = base64.b64encode(img_encoded_original).decode('utf-8')

        # Append "images" to result
        result_images.append(original_image_base64)

        output_filename = f'output{counter}{imencode_extension}'
        cv2.imwrite(os.path.join(new_folder, 'Output', output_filename), raw_image)

        os.remove(temp_img.name)
        filenames.append([input_filename, output_filename])

        counter += 1


    print('*** Running file output... ***')
    # for i in text_results:
    #     print(i)
    #     full_text = ''.join(i).lower()
    #     print(full_text)
    #     taylor_status = []

    #     for item in list_text:
    #         score = fuzz.partial_ratio(item.lower(), full_text)
    #         if score > 95:
    #             taylor_status.append(str(item))
    #         elif score > 75:
    #             taylor_status.append("Maybe " + str(item))

    #     if not taylor_status:
    #         taylor_status = 'Not detected yet.'

    #     taylor_in_text.append(str(taylor_status))
    #     print("=====================")

    # # Print the current time
    # print("current_time:", current_time)
    # print("result_images:", result_images)
    print("all_results:", all_results)
    print("text_results:", text_results)
    print("logo_bool_results:", logo_bool_results)
    print("taylor_in_text:", taylor_in_text)
    print("animation:", animation_results)

    # Open File result.json
    data = []

    for count in range(len(result_images)):
        input_abspath = os.path.abspath(f'{new_folder}/Input/{filenames[count][0]}')
        output_abspath = os.path.abspath(f'{new_folder}/Output/{filenames[count][1]}')

        # if all_results[count] == ['Not detected yet.'] and logo_bool_results[count] == 'Not detected yet.' and taylor_in_text[count] == 'Not detected yet.' and animation_results[count] == ['Not detected yet.']:
        #     image_matching_result = image_matching_function(input_abspath)
        # image_matching_result_list.append([image_matching_result])
        


        dict = {
            'input_dir': input_abspath,
            'output_dir': output_abspath,
            'celebrity': all_results[count],
            'animation': animation_results[count],
            'text': text_results[count],
            'logo': logo_bool_results[count],
            'taylor_in_text': taylor_in_text[count],
            'image_matching': image_matching_result_list[count]
        }

        data.append(dict)

    # Write the data to a JSON file
    with open(f'{new_folder}/result.json', 'w') as json_file:
        json.dump(data, json_file)

    # os.umask(prev_umask)

    # # Open File result.json
    # f = open('./result/result.json')
    # data = json.load(f)

    # for count in range(len(result_images)):
    #     dict = {
    #         'celebrity': all_results[count],
    #         'text': text_results[count],
    #         'logo': logo_bool_results[count],
    #         'taylor_in_text': taylor_in_text[count]
    #     }

    #     data.append(dict)

    # # Write the data to a JSON file
    # with open('./result/result.json', 'w') as json_file:
    #     json.dump(data, json_file)

    # if all_results[0] == []:
    #     all_results[0] = ["Not detected yet."]

    return render_template(
        'result.html',
        result_images=result_images,
        all_results=all_results,
        text_results=text_results,
        logo_bool_results=logo_bool_results,
        animation=animation_results,
        taylor_in_text=taylor_in_text,
        image_matching=image_matching_result_list
    )


# @app.route('/update_test', methods=['POST'])
# def update_test():
#     result_string = request.form
#     # for key, item in request.form.items():
#     #     result_string += f'<div>{key}: {item}</div>'
#
#     return result_string


@app.route('/update', methods=['POST'])
def update():
    class_name = request.form.get('classname')
    face_encoding_lst_add = []
    face_index_class_add = []
    files = request.files.getlist('files[]')
    bug_images = 0
    for file in files:
        temp_img = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_img.name)

        # Detect text
        img = Image.open(temp_img.name)
        temp_img.close()
        raw_image = np.array(img)
        try:
            embeddings = face_recognition.face_encodings(raw_image)[0]
            face_encoding_lst_add.append(embeddings[None])

            image_entry = dict()
            image_entry['class'] = class_name
            image_entry['filepath'] = ''

            face_index_class_add.append(image_entry)
        except:
            bug_images += 1
            print("Error!")
    known_face_encodings_list = np.load('weight/known_face_encodings_list.npy')
    with open("weight/face_index_class.pkl", 'rb') as pickle_file:
        face_index_class = pickle.load(pickle_file)

    face_encoding_lst_add = np.concatenate(face_encoding_lst_add, axis=0, dtype=np.float32)
    known_face_encodings_list = np.concatenate((known_face_encodings_list, face_encoding_lst_add), axis=0)
    face_index_class.extend(face_index_class_add)

    np.save(os.path.join('weight', 'known_face_encodings_list_temp.npy'), known_face_encodings_list)
    face_index_class_file = open(os.path.join('weight', 'face_index_class_temp.pkl'), 'wb')
    pickle.dump(face_index_class, face_index_class_file)

    return f"Classname: {class_name}\nImage added successfully but having {bug_images} images error in {len(files)} images total, redirecting...", {"Refresh": "3; url=/"}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
