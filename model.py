import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import numpy as np

"""
input: 
x_train (shape = (N1, 28, 28, 1))
y_train (shape = (N1, 10))
x_test (shape = (N2, 28, 28, 1))
y_test (shape = (N2, 10))
"""

# hyper parameter
batch_size = 256
epochs = 5

def construct_network():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))
    return model

def train(x_train, y_train, x_test, y_test):
    # model
    model = construct_network()
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                metrics=['accuracy'])

    history = model.fit(x_train, y_train,  # 画像とラベルデータ
                        batch_size=batch_size,
                        epochs=epochs,     # エポック数の指定
                        verbose=1, 
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save("mnist.h5", include_optimizer=False)

if __name__ == '__main__':
    # read dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_classes = 10
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    train(x_train, y_train, x_test, y_test)