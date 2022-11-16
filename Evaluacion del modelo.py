import tensorflow as tf
import os
from glob import glob
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from PIL import Image

fotos_mias = r'Dataset\test_set\Fotos_mias'
fotos_extraños = r'Dataset\test_set\Fotos_extraños'

fotos_mias_path = os.path.join(fotos_mias,'*')
fotos_extraños_path = os.path.join(fotos_extraños,'*')

mias_files = sorted(glob(fotos_mias_path))
extraños_files = sorted(glob(fotos_extraños_path))

n_files = len(mias_files) + len(extraños_files)
#print(n_files)

size_image = 180

allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0 

for f in mias_files:
    try:
        img = Image.open(f)
        new_img = img.resize(size=(size_image, size_image))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
        #print("Imagen cargada")
    
    except:
        #print("No cargo imagen")
        continue

for f in extraños_files:
    try:
        img = Image.open(f)
        new_img = img.resize(size=(size_image, size_image))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    
    except:
        continue

#print(allX.shape)
#print(ally.shape)

#Para cargar el modelo a evaluar:
evaluar_model = tf.keras.models.load_model('Prueba_1_fotosmias.h5')

score = evaluar_model.evaluate(allX,ally,verbose = 1)

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

pred = evaluar_model.predict(allX)
pred = np.array(pred)
label = ally[:5]

print(pred[:5])
print(label)

