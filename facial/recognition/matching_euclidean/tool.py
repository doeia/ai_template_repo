import face_recognition
import numpy
from os import listdir
from faceRecognitionSQL import FaceRecognitionSQL


class FaceRecognitionTools:
    def __init__(self):
        try:
            self.facesql = FaceRecognitionSQL()
        except:
            print("Database Connection Error")

    def encoding_FaceStr(self, image_face_encoding):
        # Convert numpy array to List
        encoding__array_list = image_face_encoding.tolist()

        # Convert the elements in the list to a string
        encoding_str_list = [str(i) for i in encoding__array_list]

        # Splice the strings in the list
        encoding_str = ','.join(encoding_str_list)

        return encoding_str

    def decoding_FaceStr(self, encoding_str):
        # print("name=%s,encoding=%s", %(name,encoding))
        # Convert the string to numpy ndarray type, that is matrix

        # Convert to a list
        dlist = encoding_str.split(' ').split(',')

        # Convert str from List to Float
        dfloat = list(map(float, dlist))

        face_encoding = numpy.array(dfloat)
        return face_encoding

    def add_Sample_Face(self, image_path):
        # Load local image file to a numpy ndarray type object
        image = face_recognition.load_image_file(image_path)
        image_face_encoding = face_recognition.face_encodings(image)[0]
        encoding_str = self.encoding_FaceStr(image_face_encoding)
        # Store face feature in the database
        sid = self.facesql.saveSampleFaceData(image_path, encoding_str)
        return sid
#########################################################################################

    def grab_Faces(self, img, model):
        faces_locations = face_recognition.face_locations(img, model=model)
        faces_encodings = face_recognition.face_encodings(
            face_image=img, known_face_locations=faces_locations)
        for idx, (face_location, face_encoding) in enumerate(zip(faces_locations, faces_encodings)):
            yield (idx+1), face_location, face_encoding
#########################################################################################

    def grab_faces_and_landmarks(self, img, model):
        faces_locations = face_recognition.face_locations(img, model=model)
        faces_encodings = face_recognition.face_encodings(
            face_image=img, known_face_locations=faces_locations)

        faces_landmarks = face_recognition.face_landmarks(face_image=img, face_locations=faces_locations
                                                          )
        for idx, (face_location, face_encoding, face_landmarks) in enumerate(zip(faces_locations, faces_encodings, faces_landmarks)):
            yield (idx+1), face_location, face_encoding, face_landmarks
#########################################################################################

    def add_Face(self, img, img_path, face_ref, face_location, face_encoding, searchFor):
        """
        Store the face encoding in a database
        """
        encoding_str = self.encoding_FaceStr(face_encoding)
        # print("Face ID=",face_ref,"Face Encoding String=",encoding_str)
        result = self.facesql.saveFaceData(img_path=img_path, face_ref=face_ref, face_location=face_location, face_encoding=self.encoding_FaceStr(image_face_encoding=face_encoding), searchFor=searchFor
                                           )
        return result
#########################################################################################

    def update_Face(self, image_name, id):
        # Load local image file to a numpy ndarray type object
        image = face_recognition.load_image_file(image_name)

        # Return 128-dimensional face encoding of each face in the image
        # There may be multiple faces in the image
        # remove the face code marked with 0 to indicate the clearest face recognized.
        image_face_encoding = face_recognition.face_encodings(image)[0]
        encoding_str = self.encoding_FaceStr(image_face_encoding)

        # Update the database of face feature encoding
        self.facesql.updateFaceData(id, encoding_str)

    def search_Face(self, id):
        face_encoding_strs = self.facesql.searchFaceData(id)

        # Face feature coding collection
        face_encodings = []

        # Face feature name collection
        face_names = []
        for row in face_encoding_strs:
            name = row[0]
            face_encoding_str = row[1]

            # Append the information obtained from the database to the collection
            face_encodings.append(self.decoding_FaceStr(face_encoding_str))
            face_names.append(name)

        return face_names, face_encodings

    def load_faceoffolder(self, folderPath):
        filename_list = listdir(folderPath)

        # Face feature coding collection
        face_encodings = []

        # Face feature name collection
        face_names = []

        a = 0
        # read the contents of the list in sequence
        for filename in filename_list:
            a += 1
            # Suffix name jpg
            if filename.endswith('jpg'):
                # Remove the last four characters of the file name to get the person name
                face_names.append(filename[:-4])

            file_str = folderPath + '/' + filename
            print("file_str = ", file_str)
            a_images = face_recognition.load_image_file(file_str)
            a_face_encoding = face_recognition.face_encodings(a_images)[0]
            face_encodings.append(a_face_encoding)
        print(face_names, a)
        return face_names, face_encodings

    def load_faceofdatabase(self):
        try:
            face_encoding_strs = self.facesql.allFaceData()
        except:
            print("Database Connection Error")

        # Face feature coding collection
        face_encodings = []
        # Face feature name collection
        face_names = []

        for row in face_encoding_strs:
            name = row[0]
            face_encoding_str = row[1]

            # Append the information obtained from the database to the collection
            face_encodings.append(self.decoding_FaceStr(face_encoding_str))
            face_names.append(name)
        return face_names, face_encodings

    def searchSimilarFaces(self, id, dist):
        results = []
        try:
            # print("FaceTools ID   = ",id)
            # print("FaceTools dist = ",dist)
            results = self.facesql.searchSimilarFaces_Man(id, dist)
        except:
            print("Database Connection Error")

        return results

    def searchSimilarFaces_Manhattan(self, id, dist):
        results = []
        try:
            # print("FaceTools ID   = ",id)
            # print("FaceTools dist = ",dist)
            results = self.facesql.searchSimilarFaces_Manhattan(id, dist)
        except:
            print("Database Connection Error")

        return results

    def searchSimilarFaces_Euclidean(self, id, dist):
        results = []
        try:
            # print("FaceTools ID   = ",id)
            # print("FaceTools dist = ",dist)
            results = self.facesql.searchSimilarFaces_Euclidean(id, dist)
        except:
            print("Database Connection Error")

        return results
