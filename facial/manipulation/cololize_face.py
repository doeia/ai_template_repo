# Import Libraries
import os
import argparse
import uuid
import mediapipe
import cv2
import filetype
import numpy as np
import config


# The colorizer model architecture
COLORIZATION_MODEL = os.path.join(
    config.MODELS_PATH, 'colorization_deploy_v2.prototxt')
# The colorizer model pre-trained weights
COLORIZATION_PROTO = os.path.join(
    config.MODELS_PATH, 'colorization_release_v2.caffemodel')
COLORIZATION_POINTS = os.path.join(config.MODELS_PATH, 'pts_in_hull.npy')

# To reduce Mediapipe false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5
EXPANDING_FACTOR = 0.35


def initialize_mediapipe():
    """
    Initializing mediapipe face detection sub-module
    """
    # Enable face detection
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(
        MIN_CONFIDENCE_LEVEL)

    return mpFaceDetection


def load_colorization_model(model, proto, pts):
    """
    Load the pre-trained colorization model
    Load the cluster center points
    Add layers to the colorization model
    """
    colorization_pts = np.load(pts)
    colorization_net = cv2.dnn.readNetFromCaffe(model, proto)
    # use CPU
    colorization_net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)

    # Add layers to the caffe model
    colorization_pts = colorization_pts.transpose().reshape(2, 313, 1, 1)
    class8_layer = colorization_net.getLayerId('class8_ab')
    conv8_layer = colorization_net.getLayerId('conv8_313_rh')

    colorization_net.getLayer(class8_layer).blobs = [
        colorization_pts.astype("float32")]
    colorization_net.getLayer(conv8_layer).blobs = [
        np.full([1, 313], 2.606, dtype="float32")]
    return colorization_net


def prepare_image(img):
    # Resize the input image to a specific width
    output = img.copy()  # imutils.resize(img,width=width)
    # Normalize the image - Scale the pixel intensities to the range [0,1]
    output_scaled = output.astype('float32') / 255.0
    output_LAB = cv2.cvtColor(output_scaled, cv2.COLOR_BGR2LAB)

    # Resize the lab frame according to the dimensions accepted by the colorization model.
    output_Resized = cv2.resize(output_LAB, (224, 224))
    # Split channels (L A B) and extract the L channel
    L = cv2.split(output_Resized)[0]
    # Perform a mean centering of the L channel.
    L -= 50
    return L


def process_image(img, colorization_net, L):
    # Pass the L channel through the network to predict the A & B chanel values.
    colorization_net.setInput(cv2.dnn.blobFromImage(L))
    # Find the A & B chanel values
    ab = colorization_net.forward()[0, :, :, :].transpose((1, 2, 0))
    # Resize the predicted ab volume to the same dimensions of the image
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    # Grab the L channel
    normalized_img = img.astype('float32')/255.0
    L = cv2.split(cv2.cvtColor(normalized_img, cv2.COLOR_BGR2LAB))[0]
    # Concatenate the L channel of the image with the predicted ab channels.
    output_colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert the image from the LAB color space to the RGB color space.
    output_colorized = cv2.cvtColor(output_colorized, cv2.COLOR_LAB2BGR)
    # Clip any values outsize the range [0,1]
    output_colorized = np.clip(output_colorized, 0, 1)
    # Convert to unsigned 8-bit integer
    output_colorized = (255*output_colorized).astype('uint8')
    return output_colorized


def enlarge_bounding_box(x, y, w, h):
    """
    Enlarge the bounding box based on the expanding factor
    """
    # create a larger bounding box with buffer around keypoints
    x1 = int(x - EXPANDING_FACTOR * w)
    w1 = int(w + 2 * EXPANDING_FACTOR * w)
    y1 = int(y - EXPANDING_FACTOR * h)
    h1 = int(h + 2 * EXPANDING_FACTOR * h)
    # print('x1,y1,w1,h1', x1, y1, w1, h1)
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    # print('x1,y1,w1,h1', x1, y1, w1, h1)
    return x1, y1, w1, h1


def colorize_faces(input_path: str, display_output: bool = False):
    """
    Colorize the faces located within a digital image.
    """
    # Initialize mediapipe face detection sub-module
    mpFaceDetection = initialize_mediapipe()

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
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

    # Initialize the colorization model
    colorization_net = load_colorization_model(model=COLORIZATION_MODEL, proto=COLORIZATION_PROTO, pts=COLORIZATION_POINTS
                                               )
    frameHeight, frameWidth, frameChannels = frame.shape
    # Convert the frame to gray format
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Loop over the faces detected
    for idx, face_detected in enumerate(faces.detections):
        # Output message
        label = f"Face ID = {(idx+1)} - Detection Score {int(face_detected.score[0]*100)}%"
        output_msg = {'msg': label, 'category': "info"}
        output_info.append(output_msg)
        print(output_msg.get('category'), output_msg.get('msg'))

        # Get the face relative bounding box
        relativeBoundingBox = face_detected.location_data.relative_bounding_box
        faceBoundingBox = int(relativeBoundingBox.xmin * frameWidth), int(relativeBoundingBox.ymin * frameHeight), int(
            relativeBoundingBox.width * frameWidth), int(relativeBoundingBox.height * frameHeight)
        # Get the coordinates of the face bounding box
        x, y, w, h = faceBoundingBox
        # Create a larger bounding box with buffer around keypoints
        x1, y1, w1, h1 = enlarge_bounding_box(x, y, w, h)

        # Extract the face -> region of interest
        roi_face_color = img[y1:y1 + h1, x1:x1 + w1]
        LChannel = prepare_image(roi_face_color)
        roi_face_colorized = process_image(
            roi_face_color, colorization_net, LChannel)

        # cv2.imshow('roi_face_colorized',roi_face_colorized)
        # cv2.waitKey(0)
        # cv2.imshow('frame',frame)
        # cv2.waitKey(0)

        if len(frame.shape) < 3:
            frame = np.stack((frame,)*3, axis=-1)

#        print('frame',frame.shape)
#        print('roi_face_colorized',roi_face_colorized.shape)

        # Restore the colorized face area
        frame[y1:y1 + h1, x1:x1 + w1] = roi_face_colorized

    if display_output:
        # Display Image on screen
        label = "Auto-Colorize Facial Photos"
        cv2.imshow(label, frame)
        # Mantain output until user presses a key
        cv2.waitKey(0)
        # Cleanup
        cv2.destroyAllWindows()
#####################
    # Save and Output the resulting image
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex)+os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, frame)
    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
        output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)

    # Save and Output the original image in gray format
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex)+os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, gray_frame)
    output_item = {'id': 2, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
        output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)
#####################
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
    colorize_faces(input_path=args['input_path'],
                   display_output=args['display_output'])
