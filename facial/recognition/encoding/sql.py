import os
import math
import sqlite3
import config


class FaceRecognitionSQL:

    def __init__(self):
        self.conn = sqlite3.connect(config.database_path)
        self.getDBVesrion()
        self.initialize_database_components()

    def getDBVesrion(self):
        cursor = self.conn.cursor()
        try:
            print("Database created and successfully accessed...")
            sqlstr = "select sqlite_version();"
            # Execute SQL statement
            cursor.execute(sqlstr)
            result = cursor.fetchall()
            print("SQLLite Version:", result)
        except Exception as e:
            print(e)
        finally:
            # Close cursor
            cursor.close()

    def initialize_database_components(self):
        cursor = self.conn.cursor()

        # Create Main Table Faces
        try:
            ################################################################################
            sqlstr = "CREATE TABLE IF NOT EXISTS Faces ("
            sqlstr += "  _ID              INTEGER PRIMARY KEY AUTOINCREMENT"
            sqlstr += " ,_PATH            VARCHAR(255) NOT NULL"
            sqlstr += " ,_FACE_REF        INTEGER DEFAULT 1"
            sqlstr += " ,_FACE_LOCATION   VARCHAR(255) NULL"
            sqlstr += " ,_ENTRY_DATE TIMESTAMP DEFAULT CURRENT_TIMESTAMP"

            for i in range(1, 129):
                sqlstr += f" ,e{i}  REAL NULL"
            sqlstr += ");"
            # Execute sql statement
            cursor.execute(sqlstr)
            ################################################################################
            sqlstr = "CREATE TABLE IF NOT EXISTS SFaces ("
            sqlstr += "  _ID              INTEGER PRIMARY KEY AUTOINCREMENT"
            sqlstr += " ,_PATH            VARCHAR(255) NOT NULL"
            sqlstr += " ,_FACE_REF        INTEGER DEFAULT 1"
            sqlstr += " ,_FACE_LOCATION   VARCHAR(255) NULL"
            sqlstr += " ,_ENTRY_DATE TIMESTAMP DEFAULT CURRENT_TIMESTAMP"

            for i in range(1, 129):
                sqlstr += f" ,e{i}  REAL NULL"
            sqlstr += ");"
            # Execute sql statement
            cursor.execute(sqlstr)
            ################################################################################
        except Exception as e:
            print(e)
        finally:
            # Close cursor
            cursor.close()

    def processFaceData(self, sqlstr, args=()):
        # print("sqlstr = ", sqlstr)

        # Use cursor() to create a cursor object
        cursor = self.conn.cursor()

        results = None
        try:
            # Execute sql statement
            cursor.execute(sqlstr, args)
            results = cursor.lastrowid

            # Submit the database
            self.conn.commit()
        except Exception as e:
            # If an error occurs, roll back and print the error message
            self.conn.rollback()
            print(e)
        finally:
            # Close cursor
            cursor.close()
        return results

    def saveFaceData(self, img_path, face_ref, face_location, face_encoding, searchFor):
        """
        Save the selected face encoding into database
        """
        imgname = img_path.split(os.path.sep)[-1]
        # print("saveFaceData Name = ", imgname)
        sqlstr = "INSERT INTO "

        if searchFor:
            sqlstr += "SFaces"
        else:
            sqlstr += "Faces"

        sqlstr += "(_PATH,_FACE_REF,_FACE_LOCATION "
        for i in range(1, 129):
            sqlstr += f",e{i}"
        sqlstr += ")"
        sqlstr += f"values('{imgname}', {face_ref}, '{face_location}', {face_encoding})"
        result = self.processFaceData(sqlstr)
        return result

    def updateFaceData(self, id, encoding_str):
        self.processFaceData(
            "update Faces set _Encoding = %s where _ID = %s", (encoding_str, id))

    def execute_float_sqlstr(self, sqlstr):
        # Use cursor() method to create a cursor object
        cursor = self.conn.cursor()

        # SQL Insert statement
        results = []
        try:
            # Execute SQL statement
            cursor.execute(sqlstr)

            # Get a list of all records
            results = cursor.fetchall()
        except Exception as e:
            # If an error occurs, rollback and print the error message
            self.conn.rollback()
            print(e)
        finally:
            cursor.close()

        return results

    def searchFaceData(self, id):
        return self.execute_float_sqlstr("select * from Faces where _ID = " + id)

    def allFaceData(self):
        return self.execute_float_sqlstr("select * from Faces order by _ID")

    def searchSimilarFaces_Manhattan(self, id, dist):
        sqlstr = " SELECT selectedFace_ID, selectedFace_PATH, savedFaces_ID, savedFaces_PATH, _Distance FROM "
        sqlstr += "("
        sqlstr += "  SELECT selectedFace._ID    selectedFace_ID"
        sqlstr += " , selectedFace._PATH  selectedFace_PATH"
        sqlstr += " , savedFaces._ID   savedFaces_ID"
        sqlstr += " , savedFaces._PATH savedFaces_PATH"
        sqlstr += " , ROUND("
        for i in range(1, 129):
            sqlstr += f" (Abs(selectedFace.e{i} - savedFaces.e{i})) " if i == 1 else f" + (Abs(selectedFace.e{i} - savedFaces.e{i})) "
        sqlstr += " ,10) _Distance"
        sqlstr += " FROM SFaces selectedFace , Faces savedFaces "
        sqlstr += "WHERE 1=1"
        sqlstr += f"  AND selectedFace._ID  = {id}"
        sqlstr += ") "
        sqlstr += f" WHERE _Distance > 0 and _Distance <= {dist}"
        sqlstr += " ORDER BY 1, 5, 3"

        # print('searchSimilarFaces_Manhattan - sqlstr = ',sqlstr)

        # Use cursor() method to create a cursor object
        cursor = self.conn.cursor()
        results = []
        try:
            # Execute SQL statement
            cursor.execute(sqlstr)

            # Get a list of all records
            results = cursor.fetchall()
        except Exception as e:
            # If an error occurs, print the error message
            print(e)
        finally:
            cursor.close()

        return results

    def searchSimilarFaces_Euclidean(self, id, dist):

        def sqlite_power(x, n):
            return pow(x, n)

        def sqlite_sqrt(x):
            return math.sqrt(x)

        sqlstr = " SELECT selectedFace_ID, selectedFace_PATH, savedFaces_ID, savedFaces_PATH, _Distance FROM "
        sqlstr += "("
        sqlstr += "  SELECT selectedFace._ID    selectedFace_ID"
        sqlstr += " , selectedFace._PATH  selectedFace_PATH"
        sqlstr += " , savedFaces._ID      savedFaces_ID"
        sqlstr += " , savedFaces._PATH    savedFaces_PATH"
        sqlstr += " , ROUND( SQRT("
        for i in range(1, 129):
            sqlstr += f" POWER(Abs(selectedFace.e{i} - savedFaces.e{i}),2) " if i == 1 else f" + POWER(Abs(selectedFace.e{i} - savedFaces.e{i}),2) "
        sqlstr += ") ,10) _Distance"
        sqlstr += " FROM sFaces selectedFace , Faces savedFaces "
        sqlstr += "WHERE 1=1"
        sqlstr += f"  AND selectedFace._ID  = {id}"
        sqlstr += ") "
        sqlstr += f" WHERE _Distance > 0 and _Distance <= {dist}"
        sqlstr += " ORDER BY 1, 5, 3"

        # print('searchSimilarFaces_Euclidean - sqlstr = ',sqlstr)

        self.conn.create_function("POWER", 2, sqlite_power)
        self.conn.create_function("SQRT", 1, sqlite_sqrt)

        # Use cursor() method to create a cursor object
        cursor = self.conn.cursor()
        results = []
        try:
            # Execute SQL statement
            cursor.execute(sqlstr)

            # Get a list of all records
            results = cursor.fetchall()
        except Exception as e:
            # If an error occurs, print the error message
            print(e)
        finally:
            cursor.close()

        return results
