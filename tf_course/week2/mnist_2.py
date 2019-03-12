#%%
import tensorflow as tf
from tensorflow import keras

class MyCallBack(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('loss') < 0.05:
      print('\nStopping Training: Reached loss of %s' % logs.get('loss'))
      self.model.stop_training = True

#%%
callbacks = MyCallBack()
mnist = keras.datasets.mnist
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

#%%
# Normalize
train_X = train_X / 255.0
test_X = test_X / 255.0

#%%
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#%%
model.fit(train_X, train_Y, epochs=4, callbacks=[callbacks])
model.evaluate(test_X, test_Y)

