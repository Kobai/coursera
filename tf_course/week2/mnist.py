#%%
import tensorflow as tf
from tensorflow import keras

#%%
mnist = keras.datasets.fashion_mnist
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

#%%
#normalize
train_X = train_X / 255.0
test_X = test_X / 255.0

#%%
# build model
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

#%%
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_Y, train_Y, epochs=1)

#%%
model.evaluate(test_X, test_Y)