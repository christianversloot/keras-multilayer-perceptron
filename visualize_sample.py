# Imports
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Configuration options
feature_vector_length = 784
num_classes = 60000

# Load the data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Visualize one sample
plt.imshow(X_train[0], cmap='Greys')
plt.show()