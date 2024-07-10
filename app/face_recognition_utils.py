import face_recognition
import cv2
import numpy as np
import math
import os


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def update_user_file(username):
    user_file_path = 'data/users.txt'
    password = "image"

    updated_lines = []
    user_found = False

    # Read existing file and update or add user
    if os.path.exists(user_file_path):
        with open(user_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                if parts[0] == username:
                    updated_lines.append(f"{username}:{password}\n")
                    user_found = True
                else:
                    updated_lines.append(line)

    if not user_found:
        updated_lines.append(f"{username}:{password}\n")

    # Write updated content back to file
    with open(user_file_path, 'w') as file:
        file.writelines(updated_lines)


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        faces_dir = 'data/faces'
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)

        for image in os.listdir(faces_dir):
            image_path = os.path.join(faces_dir, image)
            face_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_image)

            if face_encodings:
                face_encoding = face_encodings[0]
                know_face_name = os.path.splitext(image)[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(know_face_name)  # Remove file extension
                print(f"Encoded {image}")
                update_user_file(know_face_name)
            else:
                print(f"No faces found in {image}")

        print("Known faces:", self.known_face_names)

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = 'Unknown'
            confidence = 'Unknown'

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])

            self.face_names.append(f'{name} ({confidence})')

        # Draw rectangles and names
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        return frame
