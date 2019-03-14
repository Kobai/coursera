#%%
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class MyCallBack(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('loss') < 0.05:
      print('\nStopping Training: Reached loss of %s' % logs.get('loss'))
      self.model.stop_training = True

#%%
callbacks = MyCallBack()
mnist = keras.datasets.mnist
(train_X, train_Y), (test_X, test_Y) = mnist.load_data()

#%%
train_X = train_X.reshape(60000, 28, 28, 1)
train_X = train_X / 255.0
test_X = test_X.reshape(10000, 28, 28, 1)
test_X = test_X / 255.0

#%%
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#%%
model.fit(train_X, train_Y, epochs=5, callbacks=[callbacks])
test_loss, test_acc = model.evaluate(test_X, test_Y)

#%%
f, axarr = plt.subplots(3, 4)
layer_outputs = [layer.output for layer in model.layers]
activation_model = keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_X[0].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, :, :, 1], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_X[7].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, :, :, 1], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_X[26].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, :, :, 1], cmap='inferno')
  axarr[2,x].grid(False)

