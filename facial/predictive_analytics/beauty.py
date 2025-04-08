# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import config

# The gender model architecture
BEAUTY_MODEL = os.path.join(config.MODELS_PATH, 'beauty_resnet.prototxt')
# The gender model pre-trained weights
BEAUTY_PROTO = os.path.join(config.MODELS_PATH, 'beauty_resnet.caffemodel')

# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (104, 117, 123)

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


def load_caffe_models(beauty_model: str, beauty_proto: str):
    """
    load the pre-trained Caffe model for beauty estimation
    """
    beauty_net = cv2.dnn.readNetFromCaffe(beauty_model, beauty_proto)
    # use CPU
    beauty_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    return beauty_net


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


def predict_beauty(input_path: str, display_output: bool = False):
    """
    Predict the beauty of the faces showing in the image
    """
    # Initialize mediapipe face detection sub-module
    mpFaceDetection = initialize_mediapipe()

    # Load the beauty prediction model
    beauty_net = load_caffe_models(
        beauty_model=BEAUTY_MODEL, beauty_proto=BEAUTY_PROTO)

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
        blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0/255, size=(
            224, 224), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
        # Predict Beauty
        beauty_net.setInput(blob)
        beauty_preds = beauty_net.forward()
        beauty_score = round(2.0 * sum(beauty_preds[0]), 1) * 10

        # Revert back the coordinates after prediction
        startX, startY, endX, endY = x, y, (x+w), (y+h)

        # Draw the box around the detected face
        label = f"Beauty={beauty_score:.0f}%"
        print(label)
        yPos = startY - 15
        while yPos < 15:
            yPos += 15
        optimal_font_scale = get_optimal_font_scale(label, ((endX-startX)+25))
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Label the processed image
        cv2.putText(frame, label, (startX, yPos),
                    cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, (0, 255, 0), 2)
        if display_output:
            # Display processed image
            display_img("Beauty Estimator", frame)

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
    predict_beauty(input_path=args['input_path'],
                   display_output=args['display_output'])
