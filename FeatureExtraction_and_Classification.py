# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 16:38:09 2021

@author: Ramya
"""

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from skimage.io import imread, imshow
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import pandas as pd
import pickle

model = keras.models.load_model('./facenet_keras.h5')
print(model.summary())

maskon = glob('./new_dataset/maskon/*.png')
maskoff = glob('./new_dataset/maskoff/*.png')

print(len(maskon), len(maskoff))


data = maskon + maskoff

print(len(data))

embeddings = []

for path in tqdm(data):

  try:
    # Lendo e normalizando a imagem
    img = imread(path).astype('float32')/255
    
    # Aplicando um reshape na imagem para deixar o formato de acordo com o input da FaceNet
    input = np.expand_dims(img, axis=0)

    # Extraindo o vetor de embeddings
    embeddings.append(model.predict(input)[0])

  except:
    print(f'Error in {path}')
    continue

print(np.array(embeddings).shape)

print(pd.DataFrame(embeddings).head())

labels = pd.DataFrame({
    'label': [1]*len(maskon) + [0]*len(maskoff)
})

df_embeddings = pd.concat([pd.DataFrame(embeddings), labels], axis=1)
print(df_embeddings)

X = np.array(df_embeddings.drop('label', axis=1))
y = np.array(df_embeddings['label'])

print(X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

mlp_model = MLPClassifier()

mlp_model.fit(X_train, y_train)

y_pred = mlp_model.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Kappa: ', cohen_kappa_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

print(X.shape, y.shape)

mlp_model = MLPClassifier()

mlp_model.fit(X, y)


print(mlp_model.score(X, y))

pickle.dump(mlp_model, open('./mlp_model.pkl', 'wb'))