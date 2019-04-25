# ---------------------------
# Alexander Camuto, Matthew Willetts -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: acamuto@turing.ac.uk, mwilletts@turing.ac.uk
# ---------------------------
"""Training Neural Networks with Keras

Goals:
    Intro: train a neural network with high level framework Keras

Dataset:
    Digits: 10 class handwritten digits
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits

"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

# --- sklearn
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split

# --- keras
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers.core import Dense, Activation, Flatten
from keras import optimizers

if __name__ == "__main__":

    # --- Load dataset from sklearn
    digits = load_digits()

    # --- Format features and labels,
    data = np.asarray(digits.data, dtype='float32')
    target = np.asarray(digits.target, dtype='int32')

    # --- Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.15, random_state=37)

    # --- Normalize data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Reshape image to be 3D for Conv ops
    X_train = X_train.reshape(-1, 8, 8, 1)
    X_test = X_test.reshape(-1, 8, 8, 1)

    print("Training data shapes : ", X_train.shape, y_train.shape)
    print("Test data shapes : ", X_test.shape, y_test.shape)

    # To build a first neural network we need to turn the target variable into a
    # vector "one-hot-encoding" representation. Keras provides a utility function
    # to convert integer-encoded categorical variables as one-hot encoded values:

    Y_train = to_categorical(y_train)

    N = X_train.shape[1]  # input size
    H = 100  # hidden layer size or 'width'
    K = 10  # output layer size, i.e number of classes

    # --- Keras sequential model
    model = Sequential()
    model.add(Conv2D(4, 5, activation='relu', input_shape=(8, 8, 1)))
    model.add(MaxPool2D(2, strides=2))
    model.add(Flatten())
    model.add(Dense(H, input_dim=N))
    model.add(Activation("tanh"))
    model.add(Dense(K))
    model.add(Activation("softmax"))

    # --- Compile and fit the model using SGD
    model.compile(
        optimizer=optimizers.SGD(lr=0.1),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=15, batch_size=32)

    # --- Display the report for model training
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    # --- Note close the plot to keep the script running
    plt.show()

    # --- Now evaluate performance on test data:
    Y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = metrics.confusion_matrix(y_test, Y_pred)

    # --- Display confusion matrix
    df_cm = pd.DataFrame(cm, index=range(10), columns=range(10))
    ax = plt.axes()
    sn.heatmap(df_cm, annot=True, ax=ax, cmap="YlGnBu")
    ax.set_title('Test Confusion Matrix')
    plt.show()
