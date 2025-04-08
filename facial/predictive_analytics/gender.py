# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import config

# The gender model architecture
GENDER_MODEL = os.path.join(config.MODELS_PATH, 'gender_deploy.prototxt')
# The gender model pre-trained weights
GENDER_PROTO = os.path.join(config.MODELS_PATH, 'gender_net.caffemodel')

# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illumination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Represent the gender classes
GENDER_LIST = ['Male', 'Female']

# To reduce Mediapipe false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5

EXPANDING_FACTOR = 0.2


def initialize_mediapipe():
    """
    Initializing mediapipe face detection sub-module
    """
    # Enable face detection
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    return mpFaceDetection


def load_caffe_models(gender_model: str, gender_proto: str):
    """
    load the pre-trained Caffe model for gender estimation
    """
    gender_net = cv2.dnn.readNetFromCaffe(gender_model, gender_proto)
    # use CPU
    gender_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    return gender_net


def display_img(title, img):
    """
    Displays an image on screen and maintains the output until the user presses a key
    """
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setWindowTitle('img', title)
    cv2.resizeWindow('img', 600, 400)

    # Display Image on screen
    cv2.imshow('img', img)

    # Mantain output until user presses a key
    cv2.waitKey(0)

    # Destroy windows when user presses a key
    cv2.destroyAllWindows()


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


def enlarge_bounding_box(x, y, w, h):
    """
    Enlarge the bounding box based on the expanding factor
    """
    # create a larger bounding box with buffer around keypoints
    x1 = int(x - EXPANDING_FACTOR * w)
    w1 = int(w + 2 * EXPANDING_FACTOR * w)
    y1 = int(y - EXPANDING_FACTOR * h)
    h1 = int(h + 2 * EXPANDING_FACTOR * h)
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    print('x1,y1,w1,h1', x1, y1, w1, h1)
    return x1, y1, w1, h1


def predict_gender(input_path: str, display_output: bool = False):
    """
    Predict the gender of the faces showing in the image
    """
    # Initialize mediapipe face detection sub-module
    mpFaceDetection = initialize_mediapipe()

    # Load the gender prediction model
    gender_net = load_caffe_models(
        gender_model=GENDER_MODEL, gender_proto=GENDER_PROTO)

    # Read Input Image
    img = cv2.imread(input_path)

    # Copy the initial image
    frame = img.copy()

    # Convert the image from BGR to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using the rgb frame
    faces = mpFaceDetection.process(rgb_frame)

    output = []
    output_info = []

    # Output message the number of faces detected...
    output_msg = {'msg': "{} face(s) detected.".format(
        len(faces.detections)), 'category': "info"}
    output_info.append(output_msg)

    face_img = None

    # Loop over the faces detected
    for idx, face_detected in enumerate(faces.detections):
        # Output message
        label = f"Face ID = {(idx+1)} - Detection Score {int(face_detected.score[0]*100)}%"
        output_msg = {'msg': label, 'category': "info"}
        output_info.append(output_msg)
        print(output_msg.get('category'), output_msg.get('msg'))

        # Get the face relative bounding box
        relativeBoundingBox = face_detected.location_data.relative_bounding_box
        frameHeight, frameWidth, frameChannels = frame.shape
        faceBoundingBox = int(relativeBoundingBox.xmin*frameWidth), int(relativeBoundingBox.ymin*frameHeight), int(
            relativeBoundingBox.width*frameWidth), int(relativeBoundingBox.height*frameHeight)
        # Get the coordinates of the face bounding box
        x, y, w, h = faceBoundingBox

        # Enlarge the bounding box
        x1, y1, w1, h1 = enlarge_bounding_box(x, y, w, h)

        # Crop the enlarged region for better analysis
        face_img = frame[y1:y1+h1, x1:x1+w1]

        # image --> Input image to preprocess before passing it through our dnn for classification.
        # scale factor = After performing mean substraction we can optionally scale the image by some factor. (if 1 -> no scaling)
        # size = The spatial size that the CNN expects. Options are = (224*224, 227*227 or 299*299)
        # mean = mean substraction values to be substracted from every channel of the image.
        # swapRB=OpenCV assumes images in BGR whereas the mean is supplied in RGB. To resolve this we set swapRB to True.
        blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=2.0, size=(
            227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence_score = gender_preds[0][i]

        # Revert back the coordinates after prediction
        startX, startY, endX, endY = x, y, (x+w), (y+h)

        # Draw the box around the detected face
        label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
        print(label)
        yPos = startY - 15
        while yPos < 15:
            yPos += 15
        optimal_font_scale = get_optimal_font_scale(label, ((endX-startX)+50))
        box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
        cv2.rectangle(frame, (startX, startY), (endX, endY), box_color, 2)

        # Label the processed image
        cv2.putText(frame, label, (startX, yPos),
                    cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, box_color, 2)
        if display_output:
            # Display processed image
            display_img("Gender Estimator", frame)

    if display_output:
        # Cleanup
        cv2.destroyAllWindows()

    # Output image after processing
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex) + os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, frame)

    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
        output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)

    mpFaceDetection.close()
    return output_info, output


def is_valid_path(path):
    """
    Validates the path inputted and makes sure that it is a file of type image
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
    predict_gender(input_path=args['input_path'],
                   display_output=args['display_output'])
