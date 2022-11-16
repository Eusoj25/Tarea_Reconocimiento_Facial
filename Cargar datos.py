import tensorflow as tf

import pathlib
import os
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

np.set_printoptions(precision=4)

#Eliminar el doble espacio entre algunos datos de la tabla
#with open('mini_list_attr_celeba.txt','r') as f:
#    print("skipping : " + f.readline())
#    print("skipping headers : " + f.readline())
#    with open('mini_attr_celeba_prepared.txt', 'w') as newf:
#        for line in f:
#            new_line = ' '.join(line.split())
#            newf.write(new_line)
#            newf.write('\n')

df = pd.read_csv('attr_celeba_prepared.txt', sep=' ', header = None)

#print("------------")
#print(df[0].head())
#print(df.iloc[:,1:21].head())
#print("------------")

files = tf.data.Dataset.from_tensor_slices(df[0])
attribute = tf.data.Dataset.from_tensor_slices(df.iloc[:,1:].to_numpy())
data = tf.data.Dataset.zip((files,attribute))

#print(data)

path_to_image = 'img_align_celeba/'
def process_file(file_name, attribute):
    image = tf.io.read_file(path_to_image + file_name)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192,192])
    image /= 255.0
    return image, attribute

labeled_images = data.map(process_file)

#Ver Imagenes las imágenes de los rostros
for image, attri in labeled_images.take(2):
    plt.imshow(image)
    plt.show()









