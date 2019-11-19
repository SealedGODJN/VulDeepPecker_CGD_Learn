# all tutorial code are running under python3.6
# If you use the version like python2.7, please modify the code accordingly

# 5 - Classifier example

from keras.optimizers import RMSprop
from keras.layers import Dense, Activation, Bidirectional, LSTM
from keras.models import Sequential
from keras.utils import np_utils
from keras.datasets import mnist
import numpy as np
import gzip

np.random.seed(1337)

# # download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# # X shape (60,000 28*28), y shape (10,000, )
# (X_train,y_train),(X_test,y_test) = mnist.load_data()

# # To load the local datasets without redownload
# path='mnist.npz'
# f = np.load(path)
# x_train, y_train = f['x_train'], f['y_train']
# x_test, y_test = f['x_test'], f['y_test']
# f.close()

# # data pre-processing
# X_train = X_train.reshape(X_train.shape[0], -1) / 255.
# X_test = X_test.reshape(X_test.shape[0], -1) / 255.
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)

# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


# Extract the images
def extract_data(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = np.reshape(data, [num_images, -1])
    return data


def extract_labels(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data, NUM_LABELS))
        one_hot_encoding[np.arange(num_labels_data), labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding

X_train = extract_data('train-images-idx3-ubyte.gz', 60000)
y_train = extract_labels('train-labels-idx1-ubyte.gz', 60000)
X_test = extract_data('t10k-images-idx3-ubyte.gz', 10000)
y_test = extract_labels('t10k-labels-idx1-ubyte.gz', 10000)

# Another way to build your neural net
model = Sequential([
    Dense(32, input_dim=784),  # 28*28=784
    Activation('relu'),
    Dense(10),  # 10 types?
    Activation('softmax'),
])

# model = Sequential()
# model.add(Dense(units=32, input_shape=(60000,784) ))
# model.add(Dense(10, activation='softmax'))

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'mae'])

print('Training ---------')

# Another way to train the model
model.fit(X_train, y_train, epochs=2, batch_size=32)

print('\nTesting ------------')

# Evaluate the model with the metrics we defined earlier
loss, accuracy, mae = model.evaluate(X_test, y_test)

print('test loss:', loss)
print('test accuracy:', accuracy)
print('test mae:', mae)
