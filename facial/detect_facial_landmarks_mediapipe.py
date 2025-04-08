#Import Libraries
import os,argparse,uuid
import mediapipe,cv2,filetype
import config

#To reduce false results increase the confidence level
MIN_CONFIDENCE_LEVEL = 0.5

def initialize_mediapipe():
    """
    Initialize mediapipe sub-modules
    """
    #Enable the face detection sub-module.
    mpFaceDetection = mediapipe.solutions.face_detection.FaceDetection(MIN_CONFIDENCE_LEVEL)

    #Enable the face landmarks identification
    mpFaceModule = mediapipe.solutions.face_mesh

    #Enable drawing detected face landmarks over the image
    mpDrawing    = mediapipe.solutions.drawing_utils

    return mpFaceDetection, mpFaceModule,mpDrawing

def detect_facial_landmarks(input_path:str,display_output:bool = False):
    """
    Detect facial landmarks showing within the image
    """

    #Initialize mediapipe module
    mpFaceDetection,mpFaceModule,mpDrawing = initialize_mediapipe()

    #To customize how Mediapipe drawing module will draw the detected face landmarks.
    #Customizing the circles representing the landmarks
    #Thickness     -> thickness of the annotation by default 2 pixels.
    #circle_radius -> Radius of the annotation circle by default 2 pixels.
    #Color -> Color of the annotation by default green.
    cDrawingSpec = mpDrawing.DrawingSpec(thickness=1,circle_radius=2, color=(0,255,0))

    #Customizing the lines connecting the landmarks
    lDrawingSpec = mpDrawing.DrawingSpec(thickness=1, color=(0, 255, 0))

    # Read Input Image
    img = cv2.imread(input_path)

    # Preserve a copy of the original
    frame = img.copy()

    output      = []
    output_info = []

    # Convert it from BGR to RGB
    # OpenCV reads images in BGR format
    # FaceMesh accepts images in RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces     = mpFaceDetection.process(rgb_frame)

    #Initializing the FaceMesh object
    #static_image_mode -> True -> static image / False -> video stream (Defaults To False).
    #max_num_faces -> Maximum number of faces to be detected (Defaults to 1)
    #refine_landmarks -> To further refine the landmark coordinates around the eyes and lips (Default to False).
    #min_detection_confidence -> Minimum confidence value from the Face detection model (1 Successful / Default 0.5)
    #min_tracking_confidence -> ignored if static_image_mode = True (Default 0.5)

    with mpFaceModule.FaceMesh(static_image_mode=True
                              ,max_num_faces = len(faces.detections)
                              ) as mpFaceMesh:

        #Face landmarks estimation (Input image in RGB format).
        result = mpFaceMesh.process(rgb_frame)
        #print('result',result)

        #The field multi_face_landmarks contains the identified face landmarks on each detected face.
        multi_facesLandmarks = result.multi_face_landmarks
        if multi_facesLandmarks:
           #Loop through the faces
           for faceID, faceLandmarks in enumerate(multi_facesLandmarks):
               print('Face',(faceID+1))
               output_msg = {'msg': "Face {} detected with {} landmarks.". \
                   format((faceID+1), len(faceLandmarks.landmark))
                   , 'category': "info"}
               output_info.append(output_msg)

               #Drawing the landmarks
               mpDrawing.draw_landmarks(image                  = frame
                                      , landmark_list          = faceLandmarks
                                      , connections            = mpFaceModule.FACEMESH_CONTOURS
                                      , landmark_drawing_spec  = cDrawingSpec
                                      , connection_drawing_spec= lDrawingSpec
                                        )

               frame_height, frame_width, frame_channel = frame.shape

               #Loop through the detected face landmarks
               for id,landmark in enumerate(faceLandmarks.landmark):
                   #print(f'Landmark ID {id} - Position X = {landmark.x} / Y = {landmark.y} / Z = {landmark.z}')
                   relative_x,relative_y = int(landmark.x * frame_width),int(landmark.y * frame_height)
                   #print(f'Landmark ID {id} - Position X = {relative_x} / Y = {relative_y}')
                   #cv2.putText(frame,str(id),(relative_x,relative_y),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),1)
                   cv2.circle(frame, (relative_x, relative_y) , radius = 4 , color=(255, 255, 0), thickness=1)

    if display_output:
       # Display Image on screen
       label = "Facial Landmarks"
       cv2.imshow(label,frame)
       # Mantain output until user presses a key
       cv2.waitKey(0)
       # Cleanup
       cv2.destroyAllWindows()

    #Close mediapipe opened submodules
    mpFaceDetection.close()

    output_filepath = os.path.join(config.PROCESSED_PATH,str(uuid.uuid4().hex) + os.path.splitext(input_path)[1])
    cv2.imwrite(output_filepath,frame)

    output_item = {'id': 1, 'folder': config.PROCESSED_FOLDER, 'name': os.path.basename(output_filepath), 'msg': os.path.basename(output_filepath)}
    output.append(output_item)

    return output_info , output

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

    parser.add_argument('-i'
                       ,'--input_path'
                       ,dest='input_path'
                       ,type=is_valid_path
                       ,required=True
                       ,help = "Enter the path of the image file to process")

    parser.add_argument('-d'
                        , '--display_output'
                        , dest='display_output'
                        , default=False
                        , type=lambda x: (str(x).lower() in ['true', '1', 'yes'])
                        , help="Display output on screen")

    args = vars(parser.parse_args())

    #To Display The Command Line Arguments
    print("## Command Arguments #################################################")
    print("\n".join("{}:{}".format(i,j) for i,j in args.items()))
    print("######################################################################")

    return args

if __name__ == '__main__':
    # Parsing command line arguments entered by user
    args = parse_args()
    detect_facial_landmarks(input_path  = args['input_path'],display_output=args['display_output'])
