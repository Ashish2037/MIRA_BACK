import cv2
import mediapipe as mp
import utils, math
import numpy as np
from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from langchain.llms import HuggingFaceHub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from transformers import pipeline
import warnings


app = Flask(__name__)

CORS(app)





model = YOLO('yolov8s.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
              "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
              "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
              "hot dog", "pizza", "donut", "cake", "chain", "sofa", "pottedplant", "bed", "diningtable", "toilet",
              "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
              "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def rescale(img, scale=0.5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)


def detecting_mobile_and_people_count(img):
    results = model(img)
    people_count = 0
    phone = 0

    # iterating through the result
    for r in results:
        boxes = r.boxes
        # iterating through every detection one by one
        for box in boxes:
            detection =[]

            cls = int(box.cls[0])  # changing the class number from tensor to integer
            label = classNames[cls]  # retrieving the class name
            conf_score = int(box.conf[0] * 100)

            if label == 'person' and conf_score >30:
                people_count += 1
            if label == 'cell phone' and conf_score >50:
                phone +=1
            #     phone = 'Mobile Phone has been detected'
            # else:
            #     phone = " "
                
                
        if people_count > 1:
            Person = "Warning: There is more than one person"
        else:
            Person = " "
            

     
        if people_count == 0:
            face = "Face not detected"
        else:
            face = " "
        
        if phone ==0:
            mobile = ''
        else:
            mobile = 'Mobile Phone has been detected'
            
            
    return  Person, face , mobile


# variables
frame_counter = 0

# constants
FONTS = cv2.FONT_HERSHEY_COMPLEX

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# Right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

map_face_mesh = mp.solutions.face_mesh



# Landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # List of (x, y) coordinates
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # Return the list of tuples for each landmark
    return mesh_coord


# Euclidean distance
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Eyes Extractor function
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # Convert color image to grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the image
    dim = gray.shape

    # Create a mask from grayscale dimensions
    mask = np.zeros(dim, dtype=np.uint8)

    # Draw eye shape on mask with white color
    cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # Draw eyes image on the mask, where the white shape is
    eyes = cv2.bitwise_and(gray, gray, mask=mask)
    eyes[mask == 0] = 155

    # Get minimum and maximum x and y for right and left eyes
    # For Right Eye
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For Left Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # Crop the eyes from the mask
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # Return the cropped eyes
    return cropped_right, cropped_left


# Eyes Position Estimator
def positionEstimator(cropped_eye):
    # Get height and width of the eye
    h, w = cropped_eye.shape

    # Remove noise from images
    gaussian_blur = cv2.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv2.medianBlur(gaussian_blur, 3)

    # Apply thresholding to convert to a binary image
    ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)

    # Create fixed parts for the eye
    piece = int(w / 3)

    # Slice the eye into three parts
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]

    # Call pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color


# Pixel counter function
def pixelCounter(first_piece, second_piece, third_piece):
    # Count black pixels in each part
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    # Create a list of these values
    eye_parts = [right_part, center_part, left_part]

    # Get the index of the max value in the list
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ''
    if max_index == 0:
        pos_eye = "RIGHT"
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index == 2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye = "Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


@app.route('/detect_objects', methods=['POST', 'GET']) 
def detect_objects():
    print("hello")
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    
    # Read the image
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Perform object detection
    results = model(img)
    
    detected_objects = []

    Person, face , phone= detecting_mobile_and_people_count(img)

    # detected_objects.append({
                
    #             'person': Person,
    #             'face': face
    #         })

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(frame_rgb)
        result = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                # See where the user's head is tilting
                if y < -10:
                    head_pose_text = "Looking Left"
                elif y > 10:
                    head_pose_text = "Looking Right"
                elif x > 15:
                    head_pose_text = "Looking Up"
                elif x < -10:
                    head_pose_text = "Looking Down"
                else:
                    head_pose_text = "Head Forward"

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                cv2.line(image, p1, p2, (255, 0, 0), 2)

                # # Add the text on the image
                # cv2.putText(image, head_pose_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(frame, head_pose_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # detected_objects.append(head_pose_text)


        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Extract landmarks for the upper and lower lips
                upper_lip_landmarks = landmarks.landmark[0]
                lower_lip_landmarks = landmarks.landmark[87]

                # Calculate the distance between the upper and lower lip landmarks
                lip_distance = abs(upper_lip_landmarks.y - lower_lip_landmarks.y) * 100
                print(int(lip_distance))

                mouth_open = lip_distance > 3

                # Display the mouth openness status
                status_text = "Mouth Open" if mouth_open else "Mouth Closed"

            # detect_objects.append(status_text)

            # cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frame = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            # cv2.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
            #               cv2.LINE_AA)
            # cv2.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
            #               cv2.LINE_AA)

            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            eye_position, color = positionEstimator(crop_right)
            
            # utils.colorBackgroundText(frame, f'R: {eye_position}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
            eye_position_left, color = positionEstimator(crop_left)
            # utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8,8)   
           
            # detect_objects.append(eye_position,eye_position_left)
            detected_objects.append({
                'mobile':phone,
                'person': Person,
                'face': face,
                'eye_position':eye_position,
                'eye_position_left':eye_position_left,
                'status_text':status_text,
                'head_pose_text':head_pose_text

            }) 

    return jsonify({'detected_objects': detected_objects}) 

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    # Load a pre-trained sentence transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Initialize Hugging Face Hub
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.6, "max_length": 100000, "max_new_tokens": 1000})

answers =[]

@app.route('/questions', methods=['POST'])
def questions():
    global answers
    print('Hello question')
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Parse the JSON request
    data = request.json
    print(data)
    topic = data.get("domain", "")
    print(topic)

    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    

    # # Set Hugging Face Hub API token
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    # Load a pre-trained sentence transformer model
    # model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # # Initialize Hugging Face Hub
    # llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.6, "max_length": 1000000, "max_new_tokens": 1000})

    # Generate questions
    query_result = llm(f'generate 10 basic {topic} question, give only questions, questions are generated randomly, question should be diverse')

    # Split the result into a list of questions
    query = query_result.split('\n')[1:]
    query = [question.split('. ', 1)[1] for question in query]

    answers = []
    for i in range(len(query)):
        result = llm(query[i])[1:]
        answer = result.split('\n')[1:][0]
        answers.append(answer)


    return jsonify({'Question':query})




@app.route('/score', methods=['POST'])
def score():
    print('Hello, kd')

    query_result = llm(f'generate 10 basic basic question, give only questions, questions are generated randomly, question should be diverse')

    # Split the result into a list of questions
    query = query_result.split('\n')[1:]
    query = [question.split('. ', 1)[1] for question in query]

    answers = []

    for i in range(len(query)):
        result = llm(query[i])[1:]
        answer = result.split('\n')[1:][0]
        answers.append(answer)
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Parse the JSON request
    data = request.json
    flag = 0

    for i in range(len(data)):
        user_input = data[i].get("answer", "")
        print(user_input)
        if not user_input:
            return jsonify({"error": "Topic is required"}), 400
    
        embedding1 = model.encode(answers[i], convert_to_tensor=True)
        embedding2 = model.encode(user_input, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
        print(cosine_similarity)
        
        if cosine_similarity.item() > 0.5:
            flag = flag + 1
        else:
            pass

    print(flag)

    return jsonify({'SCORE':flag})
    
    
if __name__ == '__main__':
    app.run(debug=True)




