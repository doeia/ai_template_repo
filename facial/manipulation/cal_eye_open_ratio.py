# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config
from scipy.spatial import distance

# Minimum ratio determining if eyes are closed
MINIMUM_EYE_ASPECT_RATIO = 0.1

# To reduce false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5


def initialize_mediapipe():
    """
    Initializing mediapipe sub-modules
    """
    # Enable the face detection sub-module
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    # Enable the face mesh sub-module
    mpFaceModule = mediapipe.solutions.face_mesh

    return mpFaceDetection, mpFaceModule


def get_optimal_font_scale(text, width):
    """
    Determine the optimal font scale based on the hosting frame width
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        # print(new_width)
        if (new_width <= width):
            return scale/10
    return 1
#############################################################################################################


def grab_face_coordinates(img, img_landmarks):
    """
    Grab the coordinates of the face landmarks
    """
    face_coordinates = np.array([[int(min([landmark.x * img.shape[1], img.shape[1] - 1])), int(min([landmark.y * img.shape[0], img.shape[0] - 1]))]
                                 for landmark in img_landmarks.landmark]).astype(int)
    face_coordinates[face_coordinates < 0] = 0
    return face_coordinates
#############################################################################################################


def calc_right_eye_open_ratio(face_coordinates):
    """
    Calculate the right eye open ratio
    """
    # r_eye_coordinates = [face_coordinates[362]
    #                   , face_coordinates[398], face_coordinates[384], face_coordinates[385], face_coordinates[386], face_coordinates[387], face_coordinates[388], face_coordinates[466]
    #                   , face_coordinates[382], face_coordinates[381], face_coordinates[380], face_coordinates[374], face_coordinates[373], face_coordinates[390], face_coordinates[249]
    #                   , face_coordinates[263]
    #                     ]

    ew = distance.euclidean(face_coordinates[362], face_coordinates[263])
    # print('Right','ew',ew)
    eh1 = distance.euclidean(face_coordinates[398], face_coordinates[382])
    eh2 = distance.euclidean(face_coordinates[384], face_coordinates[381])
    eh3 = distance.euclidean(face_coordinates[385], face_coordinates[380])
    eh4 = distance.euclidean(face_coordinates[386], face_coordinates[374])
    eh5 = distance.euclidean(face_coordinates[387], face_coordinates[373])
    eh6 = distance.euclidean(face_coordinates[388], face_coordinates[390])
    eh7 = distance.euclidean(face_coordinates[466], face_coordinates[249])

    # Calculate the eye aspect ratio
    ear = ((eh1+eh2+eh3+eh4+eh5+eh6+eh7) / (7*ew))
    print('Right Eye - EAR', ear)
    # Calculate the eye open ratio
    eor = compute_eye_open_ratio(ear, eh1, eh2, eh3, eh4, eh5, eh6, eh7, ew)
    print('Right Eye - EOR', eor)

    return eor
##########################


def compute_eye_open_ratio(ear, eh1, eh2, eh3, eh4, eh5, eh6, eh7, ew):
    """
    Calculation formula of the eye open ration
    """
    eor = 0
    # Eye is not closed
    if ear > MINIMUM_EYE_ASPECT_RATIO:
        eor = (((eh1 + eh2 + eh3 + eh4 + eh5 + eh6 + eh7) / 7) / eh4) * 100
    return eor

##########################


def calc_left_eye_open_ratio(face_coordinates):
    """
    Calculate the left eye open ratio
    """
    # l_eye_coordinates = [face_coordinates[33]
    #                   , face_coordinates[246], face_coordinates[161], face_coordinates[160], face_coordinates[159], face_coordinates[158], face_coordinates[157], face_coordinates[173]
    #                   , face_coordinates[7]  , face_coordinates[163], face_coordinates[144], face_coordinates[145], face_coordinates[153], face_coordinates[154], face_coordinates[155]
    #                   , face_coordinates[133]
    #                     ]

    ew = distance.euclidean(face_coordinates[133], face_coordinates[33])
    # print('left ew',ew)
    eh1 = distance.euclidean(face_coordinates[246], face_coordinates[7])
    eh2 = distance.euclidean(face_coordinates[161], face_coordinates[163])
    eh3 = distance.euclidean(face_coordinates[160], face_coordinates[144])
    eh4 = distance.euclidean(face_coordinates[159], face_coordinates[145])
    eh5 = distance.euclidean(face_coordinates[158], face_coordinates[153])
    eh6 = distance.euclidean(face_coordinates[157], face_coordinates[154])
    eh7 = distance.euclidean(face_coordinates[173], face_coordinates[155])

    # Calculate the eye aspect ratio
    ear = ((eh1+eh2+eh3+eh4+eh5+eh6+eh7) / (7.0*ew))
    print('Left Eye - EAR', ear)
    # Calculate the eye open ratio
    eor = compute_eye_open_ratio(ear, eh1, eh2, eh3, eh4, eh5, eh6, eh7, ew)
    print('Left Eye - EOR', eor)

    return eor
#############################################################################################################


def draw_bounding_box(frame, label, x, y, w, h, ew, eh):
    """
    Draw bounding box around the eye considered and add a label
    """
    # create bounding box with buffer around keypoints
    eye_x1 = int(x - 0 * ew)
    eye_x2 = int((x+w) + 0 * ew)
    eye_y1 = int(y - 1 * eh)
    eye_y2 = int((y+h) + 0.75 * eh)
    # Draw a label box
    startX, startY, endX, endY = eye_x1, eye_y1, eye_x2, eye_y2
    roi_eye = frame[startY:endY, startX:endX].copy()

    yPos = startY - 15
    while yPos < 15:
        yPos += 15

    optimal_font_scale = get_optimal_font_scale(label, ((endX - startX)))

    optimal_font_scale = 1 if optimal_font_scale < 1 else optimal_font_scale
    cv2.rectangle(frame, (startX, startY), (endX, endY),
                  (0, 255, 0), 2 * round(optimal_font_scale))
    labelSize = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_DUPLEX, round(optimal_font_scale), 2)
    _x1 = startX
    _y1 = endY
    _x2 = _x1 + labelSize[0][0]
    _y2 = _y1 + int(labelSize[0][1])

    cv2.rectangle(frame, (_x1, _y1), (_x2, _y2), (0, 255, 0), cv2.FILLED)
    # Draw the label
    cv2.putText(frame, label, (_x1, _y2), cv2.FONT_HERSHEY_DUPLEX,
                int(optimal_font_scale), (0, 0, 0), 2)
    return roi_eye, frame


def draw_right_eye_bounding_box(frame, face_coordinates, label):

    r_eye_coordinates = [face_coordinates[362], face_coordinates[398], face_coordinates[384], face_coordinates[385], face_coordinates[386], face_coordinates[387], face_coordinates[388], face_coordinates[466], face_coordinates[382], face_coordinates[381], face_coordinates[380], face_coordinates[374], face_coordinates[373], face_coordinates[390], face_coordinates[249], face_coordinates[263]
                         ]
    points = np.array(r_eye_coordinates, np.int32)
    convexhull = cv2.convexHull(points)
    (x, y, w, h) = cv2.boundingRect(convexhull)

    # Calculate eye width and height
    ew = distance.euclidean(face_coordinates[362], face_coordinates[263])
    eh = distance.euclidean(face_coordinates[386], face_coordinates[374])

    roi_eye, frame = draw_bounding_box(frame, label, x, y, w, h, ew, eh)

    return roi_eye, frame


def draw_left_eye_bounding_box(frame, face_coordinates, label):

    l_eye_coordinates = [face_coordinates[33], face_coordinates[246], face_coordinates[161], face_coordinates[160], face_coordinates[159], face_coordinates[158], face_coordinates[157], face_coordinates[173], face_coordinates[7], face_coordinates[163], face_coordinates[144], face_coordinates[145], face_coordinates[153], face_coordinates[154], face_coordinates[155], face_coordinates[133]
                         ]

    points = np.array(l_eye_coordinates, np.int32)
    convexhull = cv2.convexHull(points)
    (x, y, w, h) = cv2.boundingRect(convexhull)

    # Calculate eye width and height
    ew = distance.euclidean(face_coordinates[133], face_coordinates[33])
    eh = distance.euclidean(face_coordinates[159], face_coordinates[145])
    roi_eye, frame = draw_bounding_box(frame, label, x, y, w, h, ew, eh)

    return roi_eye, frame


#############################################################################################################
def calculate_eyes_open_ratio(input_path: str, display_output: bool = False):
    """
    Calculate the eyes open ratio of the faces in the image showing in the input path
    """
    # Initialize the mediapipe sub-modules
    mpFaceDetection, mpFaceModule = initialize_mediapipe()

    # Read Input Image
    img = cv2.imread(input_path)

    # Copy the original image
    frame = img.copy()

    # Convert it to rgb format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the rgb frame
    faces = mpFaceDetection.process(rgb_frame)

    output = []
    output_info = []

    # Output a message showing the faces detected
    output_msg = {'msg': "{} face(s) detected.".format(
        len(faces.detections)), 'category': "info"}
    output_info.append(output_msg)

    mpFaceMesh = mpFaceModule.FaceMesh(
        static_image_mode=True, max_num_faces=len(faces.detections), refine_landmarks=True, min_detection_confidence=MIN_CONFIDENCE_LEVEL
    )
    landmarks = mpFaceMesh.process(rgb_frame).multi_face_landmarks

    # Loop over the faces detected
    for idx, (face_detected, face_landmarks) in enumerate(zip(faces.detections, landmarks)):
        # Output message
        label = f"Face ID = {(idx + 1)} - Detection Score {int(face_detected.score[0] * 100)}%"
        output_msg = {'msg': label, 'category': "info"}
        output_info.append(output_msg)
        print(output_msg.get('category'), output_msg.get('msg'))

        # At each iteration take a new frame copy
        frame = img.copy()

        # Determine the face bounding box
        relativeBoundingBox = face_detected.location_data.relative_bounding_box
        frameHeight, frameWidth, frameChannels = frame.shape
        faceBoundingBox = int(relativeBoundingBox.xmin*frameWidth), int(relativeBoundingBox.ymin*frameHeight), int(
            relativeBoundingBox.width*frameWidth), int(relativeBoundingBox.height*frameHeight)

        # Draw a rectangle over the face bounding box
        cv2.rectangle(frame, faceBoundingBox, (0, 255, 0), 2)
        x, y, w, h = faceBoundingBox
        # Region of interest in color mode
        roi_face_color = frame[y:y+h, x:x+w]

        # Grab the coordinate of the face landmark points
        face_coordinates = grab_face_coordinates(frame, face_landmarks)

        ######################
        # Right Eye
        right_EOR = calc_right_eye_open_ratio(face_coordinates)
        # Left Eye
        left_EOR = calc_left_eye_open_ratio(face_coordinates)
        ######################

        # Avg Eye Open Ration for both eyes
        avg_EOR = (left_EOR + right_EOR) / 2.0

        leftEyeLabel = "{:.0f}%".format(left_EOR)
        roi_left_eye, frame = draw_left_eye_bounding_box(
            frame, face_coordinates, leftEyeLabel)

        rightEyeLabel = "{:.0f}%".format(right_EOR)
        roi_right_eye, frame = draw_right_eye_bounding_box(
            frame, face_coordinates, rightEyeLabel)

        ###
        title = "Left Eye = {:.2f}%".format(left_EOR)
        title += " -- Right Eye = {:.2f}%".format(right_EOR)
        title += " -- Avg = {:.2f}%".format(avg_EOR)
        print(title)

        output_filepath = os.path.join(config.PROCESSED_PATH,
                                       str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
        cv2.imwrite(output_filepath, frame)
        output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER,
                       'name': os.path.basename(output_filepath), 'msg': title}
        output.append(output_item)
        ###

        if display_output:
            # Display Image on screen
            cv2.imshow(title, frame)
            # Mantain output until user presses a key
            cv2.waitKey(0)
            # Cleanup
            cv2.destroyAllWindows()

    mpFaceMesh.close()
    mpFaceDetection.close()
    return output_info, output


def is_valid_path(path):
    """
    Validates the path inputted and makes sure that is a file of type image
    """
    if not path:
        raise ValueError(f"Invalid Path")
    if os.path.isfile(path) and 'image' in filetype.guess(path).mime:
        return path
    else:
        raise ValueError(f"Invalid Path {path}")


def parse_args():
    """
    Get user command line parameters
    """
    parser = argparse.ArgumentParser(description="Available Options")

    parser.add_argument('-i', '--input_path', dest='input_path', type=is_valid_path,
                        required=True, help="Enter the path of the image file to process")

    parser.add_argument('-d', '--display_output', dest='display_output', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help="Display output on screen")

    args = vars(parser.parse_args())

    # To Display The Command Line Arguments
    print("## Command Arguments #################################################")
    print("\n".join("{}:{}".format(i, j) for i, j in args.items()))
    print("######################################################################")

    return args


if __name__ == '__main__':
    # Parsing command line arguments entered by user
    args = parse_args()
    calculate_eyes_open_ratio(
        input_path=args['input_path'], display_output=args['display_output'])
