# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:46:31 2021

@author: Ramya
"""

import pickle
from PIL import Image
from mtcnn import MTCNN
from tensorflow import keras
import cv2 as cv
import numpy as np

facenet_model = keras.models.load_model('./facenet_keras.h5')
mlp_model = pickle.load(open('./mlp_model.pkl', 'rb'))
detector = MTCNN()

label_description = {
    0: 'NO MASK',
    1: 'MASK'
}

color = {
    0: (0, 0, 255),
    1: (0, 255, 0)
}

def get_faces(image, size=(160, 160)):
    img = np.asarray(image)
    results = detector.detect_faces(np.asarray(image))
    faces = []
    for i in range(len(results)):
      try:
        if results[i]['confidence'] > 0.95:
          x1, y1, w, h = results[i]['box']
          x2, y2 = x1 + w, y1 + h
          face = image[y1:y2, x1:x2]
          faces.append({
              'x1': x1,
              'y1': y1,
              'x2': x2,
              'y2': y2,
              'face': np.array(Image.fromarray(face).resize(size)),
              'confidence': results[i]['confidence']
          })

      except:
        continue
    return faces

def get_embeddings(face, facenet_model):
  img = np.array(face).astype('float32')/255
  input = np.expand_dims(img, axis=0)
  embedding = facenet_model.predict(input)[0]
  return embedding

def get_label(embedding, mlp_model):
    embedding = np.expand_dims(embedding, axis=0)
    proba = mlp_model.predict_proba(embedding)
    label = np.argmax(proba)
    if proba[0][label] < 0.95:
        label = abs(label-1)
    return label

def mark_points_in_frame(frame):
    img = np.asarray(frame)
    faces = get_faces(img)
    no_mask_count = 0
    for face in faces:
        x1 = face['x1']
        x2 = face['x2']
        y1 = face['y1']
        y2 = face['y2']
        emb = get_embeddings(face['face'], facenet_model)
        label = get_label(emb, mlp_model)
        if not label: no_mask_count += 1
        font = cv.FONT_HERSHEY_TRIPLEX
        font_scale = 0.5
        img = cv.rectangle(img, (x1, y1), (x2, y2), color[label], 2)
        cv.putText(img, label_description[label], (x1, y1-10), font, fontScale=font_scale,
                    color=color[label], thickness=1)
        cv.putText(img, f'People without mask: {no_mask_count}', (15, 15), font, fontScale=0.6,
                    color=(0, 0, 0), thickness=1)
    return img, no_mask_count

def start_capture():
    #print('Starting capture...\n')
    vid = cv.VideoCapture(0)
    #print('Capture started!')
    iter_count=0
    while (True):
        iter_count = iter_count + 1
        if iter_count==10:
            break
        _, frame = vid.read() 
        detec, count = mark_points_in_frame(frame) 
        cv.imshow('frame', detec) 
        if cv.waitKey(1) & 0xFF == ord('q'): 
            break 
        if(count>0):
            ret_val=False
        else: 
            ret_val=True
    iter_count=0
    vid.release() 
    cv.destroyAllWindows() 
    del (vid)
    return ret_val

ret_val= start_capture()
print(ret_val)