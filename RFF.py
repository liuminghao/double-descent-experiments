import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

 (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")


model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    RandomFourierFeatures(
        output_dim=4000,
        kernel_initializer='gaussian'),
    layers.Dense(10),
    layers.Dense(10, activation='softmax'),
])
model.summary()
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['categorical_accuracy']
)
y_train_one_hot = tf.one_hot(y_train, depth=10)
y_test_one_hot = tf.one_hot(y_test, depth=10)

x_train_sampled = x_train[:10000]
y_train_sampled = y_train_one_hot[:10000]

history = model.fit(x_train_sampled, y_train_sampled, epochs=50)

example = 30
model.predict(x_test)[example]
plt.imshow(x_test[example])
y_test[example]
model.evaluate(x_test, y_test_one_hot)
