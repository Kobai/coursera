#%%
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%%
# root = <project directory>
human_dir = os.path.join(root, 'horse-or-human/humans')
horse_dir = os.path.join(root, 'horse-or-human/horses')

train_humans = os.listdir(human_dir)
train_horses = os.listdir(horse_dir)

#%%
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

#%%
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

#%%
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
  os.path.join(root,'horse-or-human'),
  target_size=(300, 300),
  batch_size=128,
  class_mode='binary'
)

#%%
history = model.fit_generator(
  train_generator,
  steps_per_epoch=8,
  epochs=15,
  verbose=1
)

#%%
import numpy as np
from tensorflow.keras.preprocessing import image

img = image.load_img(os.path.join(root,'unicorn.png'), target_size=(300,300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
if classes[0] > 0.5:
  print('human')
else:
  print('horse')



