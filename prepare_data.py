import os
import cv2
import numpy as np
import pickle
from os import listdir
from os.path import isdir
from PIL import Image
import matplotlib.pyplot as plt
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import Normalizer



# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image_rgb = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]


# load the facenet model
facenet_model = load_model('Data/model/facenet_keras.h5', compile=False)

people_dir = 'Data/People'
encodings_path = 'Data/Encoding/encodings.pkl'

encoding_dict = dict()

for person_name in os.listdir(people_dir):
    person_dir = os.path.join(people_dir, person_name)
    encodes = []
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        result = extract_face(img_path)
        encode = get_embedding(facenet_model,result)
        encodes.append(encode)

        if encodes:
            encode = np.sum(encodes, axis=0)
            encode = Normalizer(norm='l2').fit_transform(np.expand_dims(encode, axis=0))[0]
            encoding_dict[person_name] = encode

for key in encoding_dict.keys():
    print(key)

with open(encodings_path, 'bw') as file:
    pickle.dump(encoding_dict, file)
