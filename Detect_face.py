import mtcnn
import cv2
import pickle
import numpy as np
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
from keras.models import load_model
from statsmodels.iolib import load_pickle
from PIL import Image




def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = np.mean(face_pixels), np.std(face_pixels)
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


# load the facenet model
facenet_model = load_model('Data/model/facenet_keras.h5', compile=False)

face_detector = mtcnn.MTCNN(min_face_size=50)
conf_t = 0.99
vc = cv2.VideoCapture(0)

while vc.isOpened():
    ret, frame = vc.read()
    if not ret:
        print(":(")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(frame_rgb)

    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = frame_rgb[y1:y2, x1:x2]
    face = Image.fromarray(face)
    face = face.resize((160, 160))
    face_array = np.asarray(face)
    encode = get_embedding(facenet_model, face_array)
    encode = Normalizer(norm='l2').fit_transform(encode.reshape(1, -1))[0]

    name = 'unknown'

    encodings_path = 'Data/Encoding/encodings.pkl'
    encoding_dict = load_pickle(encodings_path)

    recognition_t = 0.5

    distance = float("inf")
    for db_name, db_encode in encoding_dict.items():
        dist = cosine(db_encode, encode)
        if dist < recognition_t and dist < distance:
            name = db_name
            distance = dist

    for res in results:
        x1, y1, width, height = res['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        confidence = res['confidence']
        if confidence < conf_t:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 0), thickness=4)
        if name == 'unknown':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 0), thickness=4)
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 0), thickness=2)
            cv2.putText(frame, name, (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Face Recogition Using FaceNet", frame)

    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break


vc.release()

# Closes all the frames
cv2.destroyAllWindows()

