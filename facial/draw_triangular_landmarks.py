# Import Libraries
import os
import argparse
import uuid
import dlib
import cv2
import filetype
import numpy as np
import config
import mediapipe

# Landmark's facial detector to estimate the location of 68 coordinates that map the facial points
# in a person's face
FACIAL_LANDMARK_PREDICTOR = os.path.join(
    config.MODELS_PATH, 'shape_predictor_68_face_landmarks.dat')
# To reduce false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5


def initialize_dlib(facial_landmark_predictor: str):
    """
    Initialize dlib's face detetctor (HOG-based) and then create the facial landmark predictor
    """
    print('Loading facial landmark predictor...')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(FACIAL_LANDMARK_PREDICTOR)

    return detector, predictor


def initialize_mediapipe():
    """
    Initializing mediapipe sub-modules
    """
    # Enables face detection and landmarks identification
    mpFaceModule = mediapipe.solutions.face_mesh

    return mpFaceModule


def extract_index_nparray(nparray):
    """
    # Extracting index from a numpy array
    """
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def segment_into_triangles(landmarks_points):
    """
    Mesh up the face landmark points into triangles.
    Devise a convex hull representing the boundary of the collection of landmark points
    Perform a Delaunay Triangulation of the landmark points
    """
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    # Delaunay triangulation

    # Draw a bounding rectangle around the convex hull
    rect = cv2.boundingRect(convexhull)

    # Create an instance of Subdiv2D which subdivides a plane into triangles using the Delaunary's algorithm.
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    indexes_triangles = []
    # Loop over the list of triangles
    # Get the landmark points indexes of each triangle.
    for t in triangles:
        # point 1
        pt1 = (t[0], t[1])
        # point 2
        pt2 = (t[2], t[3])
        # point 3
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)
        if index_pt1 is not None and \
           index_pt2 is not None and \
           index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
    return indexes_triangles


def draw_triangles(src_img, indexes_triangles, landmarks_points):
    """
    Loop over the triangles and draw their vertices.
    """
    img = src_img.copy()
    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
        cv2.drawContours(img, [triangle], 0, (0, 255, 0), 1)
    return img
####################################################################################################################


def draw_dlib_triangles(face, frame, gray_frame, predictor):
    # Determine the facial landmarks for the face region
    # Convert the facial landmarks to a list
    shape = predictor(gray_frame, face)

    landmarks_points = []
    for n in range(0, 68):
        x = shape.part(n).x
        y = shape.part(n).y
        landmarks_points.append((x, y))

    # print('DLIB', len(landmarks_points))

    # Delaunay triangulation
    img_indexes_triangles = segment_into_triangles(landmarks_points)
    # Drawing triangles
    trg_img = draw_triangles(frame, img_indexes_triangles, landmarks_points)
    return trg_img, len(landmarks_points)


def draw_mp_triangles(frame, faces):
    # Mediapipe Delaunay Triangulation
    # Convert the input image from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Initialize mediapipe sub-module
    mpFaceModule = initialize_mediapipe()

    mpFaceMesh = mpFaceModule.FaceMesh(
        static_image_mode=True, max_num_faces=len(faces), refine_landmarks=True, min_detection_confidence=MIN_CONFIDENCE_LEVEL
    )

    landmarks = mpFaceMesh.process(rgb_frame).multi_face_landmarks
    if landmarks:
        frame_height, frame_width, frame_channel = rgb_frame.shape
        trg_img = frame.copy()
        # Loop through the faces
        for faceID, faceLandmarks in enumerate(landmarks):
            landmarks_points = []
            # Loop through the detected face landmarks
            for id, landmark in enumerate(faceLandmarks.landmark):
                relative_x, relative_y = int(
                    landmark.x * frame_width), int(landmark.y * frame_height)
                landmarks_points.append((relative_x, relative_y))

            # print('Mediapipe', len(landmarks_points))
            # Segment into triangles
            img_indexes_triangles = segment_into_triangles(landmarks_points)
            # Draw the triangles
            trg_img = draw_triangles(
                trg_img, img_indexes_triangles, landmarks_points)

    mpFaceMesh.close()
    return trg_img, len(landmarks_points)


####################################################################################################################
def draw_triangular_landmarks(input_path: str, display_output: bool = False):
    """
    Draw triangular facial landmarks of the faces showing within the image
    """
    # Initialize dlib face detector
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

###################################################################################################
# DLIB Delaunay Triangulation
    dlib_frame = frame.copy()
    # Loop over the faces detected
    for idx, face in enumerate(faces):
        # Output message
        output_msg = {'msg': "Face {} detected on position (Left:{} Top:{} Right:{} Botton:{}).".
                      format((idx+1), face.left(), face.top(), face.right(), face.bottom()), 'category': "info"}
        output_info.append(output_msg)
        print(output_msg.get('category'), output_msg.get('msg'))

        # Draw Delaunay Triangulation based on Dlib library
        dlib_frame, len_landmarks_points = draw_dlib_triangles(
            face, dlib_frame, gray_frame, predictor)

    # Output Processed Image
    dlib_label = f'Delaunay Landmarking using Dlib - {len_landmarks_points} points'
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex) + os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, dlib_frame)
    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER,
                   'name': os.path.basename(output_filepath), 'msg': dlib_label}
    output.append(output_item)
###################################################################################################
# Mediapipe Delaunay Triangulation
    mp_frame = frame.copy()

    # Draw Delaunay Triangulation based on MediaPipe library
    mp_frame, len_landmarks_points = draw_mp_triangles(mp_frame, faces)
    # Output Processed Image
    mp_label = f'Delaunay Landmarking using MediaPipe - {len_landmarks_points} points'
    output_filepath = os.path.join(config.PROCESSED_PATH, str(
        uuid.uuid4().hex) + os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath, mp_frame)
    output_item = {'id': 2, 'folder': config.PROCESSED_FOLDER,
                   'name': os.path.basename(output_filepath), 'msg': mp_label}
    output.append(output_item)
###################################################################################################
    if display_output:
        # Display Image on screen
        cv2.imshow(dlib_label, dlib_frame)
        # Maintain output until user presses a key
        cv2.waitKey(0)

        cv2.imshow(mp_label, mp_frame)
        cv2.waitKey(0)
        # Cleanup
        cv2.destroyAllWindows()

    return output_info, output
#####################################################################################################################


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
    draw_triangular_landmarks(
        input_path=args['input_path'], display_output=args['display_output'])
