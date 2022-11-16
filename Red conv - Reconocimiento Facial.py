import tensorflow as tf

import datetime
import pathlib
import os
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import keras
from keras import losses
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop, Adam, Adamax

np.set_printoptions(precision=4)

batch_size = 50
img_height = 180
img_width = 180

#Cargamos los datos y cambios nuestras etiquetas con valor -1 a 0
df = pd.read_csv('mini_attr_celeba_prepared.txt', sep=' ', header = None)
df = df.replace(-1,0)

files = tf.data.Dataset.from_tensor_slices(df[0])
attribute = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files,attribute))

path_to_image = 'mini_img_align_celeba/'
def process_file(file_name, attribute):
    image = tf.io.read_file(path_to_image + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_width,img_height])
    image /= 255.0
    return image, attribute

#Se crea un conjunto de datos de pares de image, attribute
AUTOTUNE = tf.data.AUTOTUNE
labeled_images = data.map(process_file,num_parallel_calls=AUTOTUNE)

#Se divide el conjunto de datos en conjuntos de entrenamiento y validación
image_count = len(labeled_images)
val_size = int(image_count * 0.2)
train_ds = labeled_images.skip(val_size)
val_ds = labeled_images.take(val_size)

#print(tf.data.experimental.cardinality(train_ds).numpy())
#print(tf.data.experimental.cardinality(val_ds).numpy())

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

#Construir y entrenar un modelo
num_classes = 40

#Para cargar la red:
modelo_cargado = tf.keras.models.load_model('Prueba_1.h5')

#model = tf.keras.Sequential([
#  tf.keras.layers.Conv2D(20, 4, input_shape = (img_width,img_height,3), activation='relu',),
#  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#  tf.keras.layers.Conv2D(20, 4, activation='relu'),
#  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#  tf.keras.layers.Conv2D(20, 4, activation='relu'),
#  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#  tf.keras.layers.Conv2D(20, 4, activation='relu'),
#  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#  tf.keras.layers.Flatten(),
#  tf.keras.layers.Dense(128, activation='relu'),
#  tf.keras.layers.Dense(num_classes, activation='sigmoid')
#])

modelo_cargado.summary()

modelo_cargado.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['binary_accuracy'])

log_dir="Graph/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
#python -m tensorboard.main --logdir=/Graph  <- Para correr Tensor board
#tensorboard  --logdir Graph/

modelo_cargado.fit(
    train_ds,
    batch_size=batch_size,
    epochs=15,
    verbose=1,
    validation_data=val_ds,
    callbacks= [tbCallBack])

#Para guardar el modelo en disco
modelo_cargado.save("Prueba_2.h5")


