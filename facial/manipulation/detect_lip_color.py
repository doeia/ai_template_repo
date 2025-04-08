# Import Libraries
import os
import argparse
import uuid
import dlib
import cv2
import filetype
import numpy as np
from imutils import face_utils
import config
import webcolors
from sklearn.cluster import KMeans
from collections import Counter

# Landmark's facial detector to estimate the location of 68 coordinates that map the facial points
# in a person's face
FACIAL_LANDMARK_PREDICTOR = os.path.join(
    config.MODELS_PATH, 'shape_predictor_68_face_landmarks.dat')


def initialize_dlib(facial_landmark_predictor: str):
    """
    Initialize dlib's face detetctor (HOG-based) and then create the facial landmark predictor
    """
    print('Loading facial landmark predictor...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

    return detector, predictor


def remove_black_areas(estimator_labels, estimator_cluster):
    """
    Remove out the black pixel from skin area extracted
    By default OpenCV does not handle transparent images and replaces those with zeros (black).
    Useful when thresholding is used in the image.
    """
    # Check for black
    hasBlack = False

    # Get the total number of occurence for each color
    occurence_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurence_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurence
            del occurence_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurence_counter, estimator_cluster, hasBlack)


def get_color_information(estimator_labels, estimator_cluster, hasThresholding=False):
    """
    Extract color information based on predictions coming from the clustering.
    Accept as input parameters estimator_labels (prediction labels)
                               estimator_cluster (cluster centroids)
                               has_thresholding (indicate whether a mask was used).
    Return an array the extracted colors.
    """
    # Variable to keep count of the occurence of each color predicted
    occurence_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurence, cluster, black) = remove_black_areas(
            estimator_labels, estimator_cluster)
        occurence_counter = occurence
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurence_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurences
    totalOccurence = sum(occurence_counter.values())

    # Loop through all the predicted colors
    for x in occurence_counter.most_common(len(estimator_cluster)):
        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index - 1) if ((hasThresholding & hasBlack)
                                & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1] / totalOccurence)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extract_dominant_colors(image, number_of_colors=5, hasThresholding=False):
    """
    Putting all together.
    Accept as input parameters image -> the input image in BGR format (8 bit / 3 channel)
                                     -> the number of colors to extracted.
                                     -> hasThresholding indicate whether a thresholding mask was used.
    Leverage machine learning by using an unsupervised clustering algorithm (Kmeans Clustering) to cluster the
    image pixels data based on their RGB values.
    """
    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colors Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0] * img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Color Information
    colorInformation = get_color_information(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def get_top_dominant_color(dominant_colors):
    def find_closest_color(req_color):
        # This is the function which converts an RGB pixel to a color name
        min_colors = {}
        for name, key in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(name)
            rd = (r_c - req_color[0]) ** 2
            gd = (g_c - req_color[1]) ** 2
            bd = (b_c - req_color[2]) ** 2
            min_colors[(rd + gd + bd)] = key
            closest_name = min_colors[min(min_colors.keys())]
        return closest_name

    # print(dominant_colors[0].get('cluster_index'))
    # print(dominant_colors[0].get('color'))
    # print(dominant_colors[0].get('color_percentage'))

    color_value = (
        int(dominant_colors[0].get('color')[2]), int(dominant_colors[0].get(
            'color')[1]), int(dominant_colors[0].get('color')[0])
    )
    closest_color_name = find_closest_color(
        (
            int(dominant_colors[0].get('color')[0]), int(dominant_colors[0].get(
                'color')[1]), int(dominant_colors[0].get('color')[2])
        )
    )
    color_score = round(dominant_colors[0].get('color_percentage') * 100, 2)
    return color_value, closest_color_name, color_score


def process_lips_area(shape, img, mouth_landmark):
    """
    Create a mask for the lips region
    localize the lips region
    Get the top dominant color of the lips region
    """
    name, (i, j) = mouth_landmark
    pts = np.array([shape[i:j]])

    # Extract the ROI of the face region as a separate image
    (x, y, w, h) = cv2.boundingRect(pts)
    roi = img[y:y + h, x:x + w].copy()

    # Create a mask
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    mask = mask[y:y + h, x:x + w].copy()

    # Lips Region of interest
    masked_img = cv2.bitwise_and(roi, roi, mask=mask)

    dominant_colors = extract_dominant_colors(masked_img, hasThresholding=True)
    top_color_value, top_color_name, top_color_score = get_top_dominant_color(
        dominant_colors)
    print(
        f'Top Color Value {top_color_value}, Name {top_color_name}, Score {top_color_score}')

    # top_color_name,top_color_score = top_image_colors(new_img, 1)
    return masked_img, top_color_name, top_color_score


def calculate_optimal_fontscale(text, width):
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


def recognize_lips_color(input_path: str, display_output: bool = False):
    """
    Recognize the color and shape of the lips of the faces showing within a digital image
    """
    # Initialize dlib face detector using the facial landmark recognition
    detector, predictor = initialize_dlib(
        facial_landmark_predictor=FACIAL_LANDMARK_PREDICTOR)

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()

    # Convert it to gray scale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray scale frame
    faces = detector(gray_frame, 0)

    output = []
    output_info = []

    # Loop over the faces detected
    for idx, face in enumerate(faces):
        # At each iteration take a new frame copy
        frame = img.copy()

        output_msg = {'msg': "Face {} detected on position (Left:{} Top:{} Right:{} Botton:{}).".
                      format((idx+1), face.left(), face.top(), face.right(), face.bottom()), 'category': "info"}
        output_info.append(output_msg)
        print(output_msg.get('category'), output_msg.get('msg'))

        # Determine the facial landmarks for the face region
        # Convert the facial landmarks to a Numpy Array
        shape = predictor(gray_frame, face)
        shape = face_utils.shape_to_np(shape)

        # List containing the facial features
        face_landmarks = list(face_utils.FACIAL_LANDMARKS_IDXS.items())

        # The 8 landmarks of the face
        # Facial features were detected
        if (len(face_landmarks)):
            # Devise a face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(face)
            startX, startY, endX, endY = x, y, (x + w), (y + h)

            # The mouth landmark
            # Represents the mouth face_landmarks[0]
            masked_img, top_color_name, top_color_score = process_lips_area(
                shape, frame, face_landmarks[0])

            label = "{}-{:.2f}%".format(top_color_name, top_color_score)
            print(label)

            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
######################
            # Output the processed frame
            output_filepath = os.path.join(config.PROCESSED_PATH,
                                           str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
            cv2.imwrite(output_filepath, frame)
            output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
                output_filepath), 'msg': os.path.basename(output_filepath)}
            output.append(output_item)

            # Output the lips shape and display their top dominant color
            output_filepath = os.path.join(config.PROCESSED_PATH,
                                           str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
            cv2.imwrite(output_filepath, masked_img)
            output_item = {'id': 2, 'folder': config.PROCESSED_FOLDER,
                           'name': os.path.basename(output_filepath), 'msg': label}
            output.append(output_item)
######################
            if display_output:
                # Display Image on screen
                label = "Lips Color Recognizer"
                # Maintain output until user presses a key
                cv2.imshow(label, frame)
                cv2.waitKey(0)
                cv2.imshow(label, masked_img)
                cv2.waitKey(0)

        if display_output:
            # Cleanup
            cv2.destroyAllWindows()

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
    recognize_lips_color(
        input_path=args['input_path'], display_output=args['display_output'])
