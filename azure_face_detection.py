import requests

def azure_face_recognition(image_path):
    # API endpoint
    url = ""

    # Headers
    headers = {
        "Host": "eastus.api.cognitive.microsoft.com",
        "Content-Type": "application/octet-stream",  # Use binary data
        "Ocp-Apim-Subscription-Key": "7857e53d930f42b0b7c25d9ccf3b9db7"
    }

    # Read the image as binary data
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Send POST request with binary data
    response = requests.post(url, headers=headers, data=image_data)

    # List of faces:
    list_faces = []

    # Check the response
    if response.status_code == 200:
        # Parse and print the response content
        result = response.json()
        
        for face in result:
            # print("Face ID:", face["faceId"])
            # print("Face rectangular: ", face["faceRectangle"])
            list_faces.append(face["faceRectangle"])

        return list_faces
    else:
        # Print the error message
        print("Error:", response.status_code, response.text)

        return []   
