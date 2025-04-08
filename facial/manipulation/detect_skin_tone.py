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

# Defining the HSV Thresholds representing the HSV pixel intensities to be considered skin
LOWER_THRESHOLD = np.array([0, 48, 80], dtype=np.uint8)
UPPER_THRESHOLD = np.array([20, 255, 255], dtype=np.uint8)


def initialize_dlib(facial_landmark_predictor: str):
    """
    Initialize dlib's face detetctor (HOG-based) and then create the facial landmark predictor
    """
    print('Loading facial landmark predictor...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

    return detector, predictor


"""
def top_image_colors(img,n):
    import pandas as pd
    #Convert the image to RGB
    image = img.convert('RGB')

    detected_colors = []
    for x in range(image.width):
        for y in range(image.height):
            (a,b,c) = image.getpixel((x,y))
            if a != 0 and b!=0 and c!=0:
                detected_colors.append(find_color( (a,b,c) ))
    Series_Colors = pd.Series(detected_colors)
    output = Series_Colors.value_counts()/len(Series_Colors)
    l_colors = output.reset_index().values.tolist()
    top_color_name  = l_colors[0][0]
    top_color_score = round(l_colors[0][1] * 100 ,0)
    return top_color_name, top_color_score
"""


def extract_faces_landmarks(img, detector, predictor):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the gray scale frame
    faces = detector(img_gray, 0)

    for idx, face in enumerate(faces):
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        yield {
            "face": face, "landmarks": landmarks_points
        }


def mask_landmark(img, pts):
    # Create a mask
    mask = np.ones(img.shape[:2], np.uint8)  # np.zeros(img.shape[:2],np.uint8)
    cv2.drawContours(mask, [pts], -1, (0, 0, 0), -1, cv2.LINE_AA)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return masked_img


def extract_face_skin_area(img, landmarks_points, shape, face_landmarks, face):
    # print('landmarks', landmarks_points)
    # print('shape', shape)
    # print('face_landmarks', face_landmarks)
    # print('face', face)

    # Convert image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_mask = np.zeros_like(img_gray)
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    head_mask = cv2.fillConvexPoly(face_mask, convexhull, 255)
    masked_img = cv2.bitwise_and(img, img, mask=head_mask)

    for name, (i, j) in (
        face_landmarks[0]  # mouth
            , face_landmarks[1]  # inner_mouth
            , face_landmarks[2]  # right_eyebrow
            , face_landmarks[3]  # left_eyebrow
            , face_landmarks[4]  # right_eye
            , face_landmarks[5]  # left_eye
    ):
        pts = np.array([shape[i:j]])
        masked_img = mask_landmark(masked_img, pts)

    (x, y, w, h) = face_utils.rect_to_bb(face)
    print('x, y, w, h', x, y, w, h)
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y
    masked_face = masked_img[y:y + h, x:x + w]
    return masked_face


def threshold_face_skin_area(img):
    """
    Perform thresholding based on the range of the thresholds specified to extract pixels
    that corresponds to the skin color range.
    Take an 8 bit 3 channel image in BGR colorspace and returns the extracted image in the
    same colorspace.
    """
    # Take a copy of the image
    img = img.copy()

    # Converting from BGR Colors Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Single Channel mask,denoting presence of colors in the about threshold
    skinMask = cv2.inRange(img, LOWER_THRESHOLD, UPPER_THRESHOLD)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


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


"""
def prety_print_data(color_info):
  for x in color_info:
    print(pprint.pformat(x))
    print()

def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0), (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar
"""


def detect_skin_color(input_path: str, display_output: bool = False):
    """
    Detect Face Skin Color
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

    output = []
    output_info = []

    # Loop over the faces detected
    for idx, face_landmarks in enumerate(extract_faces_landmarks(frame, detector, predictor)):
        frame = img.copy()

        face = face_landmarks['face']
        landmark_points = face_landmarks['landmarks']
        # print('landmark_points', landmark_points)

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

        # Draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face)
        startX, startY, endX, endY = x, y, (x + w), (y + h)

        # Extract Face Skin Area
        face_skin_area = extract_face_skin_area(
            frame, landmark_points, shape, face_landmarks, face)

        # Threshold Skin Area
        thresholded_skin = threshold_face_skin_area(face_skin_area)

        dominant_colors = extract_dominant_colors(
            thresholded_skin, number_of_colors=5, hasThresholding=True)

        color_value, closest_color_name, color_score = get_top_dominant_color(
            dominant_colors)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color_value, 5)

        label = "{}-{:.2f}%".format(closest_color_name, color_score)
        print(label)
######################
        output_filepath = os.path.join(config.PROCESSED_PATH,
                                       str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
        cv2.imwrite(output_filepath, frame)
        output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(
            output_filepath), 'msg': os.path.basename(output_filepath)}
        output.append(output_item)

        output_filepath = os.path.join(config.PROCESSED_PATH,
                                       str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
        cv2.imwrite(output_filepath, thresholded_skin)
        output_item = {'id': 2, 'folder': config.PROCESSED_FOLDER,
                       'name': os.path.basename(output_filepath), 'msg': label}
        output.append(output_item)
######################

        if display_output:
            # Display Image on screen
            cv2.imshow(label, frame)
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
    detect_skin_color(
        input_path=args['input_path'], display_output=args['display_output'])
