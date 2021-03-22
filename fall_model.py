from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Convolution3D, Conv3D, MaxPooling3D, BatchNormalization, Flatten, Conv2D, \
    MaxPooling2D, TimeDistributed
from keras.optimizers import SGD
import matplotlib.pyplot as plt


def baseline_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(Dense(input_dim=x_train.shape[1], activation='sigmoid', units=100))
    model.add(Dense(activation='relu', units=100))
    model.add(Dropout(0.1))
    model.add(Dense(activation='relu', units=100))
    model.add(Dropout(0.1))
    model.add(Dense(activation='softmax', units=3))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=60)
    result = model.evaluate(x_test, y_test)
    print(result)
    model.save("fall_baseline_model.h5")


def draw_history(history):
    plt.plot(history.history['loss'], color='r')
    plt.plot(history.history['val_loss'], color='g')
    plt.plot(history.history['accuracy'], color='b')
    plt.plot(history.history['val_accuracy'], color='k')
    plt.title('Model Acc and Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'test_loss', 'train_acc', 'test_acc'], loc='upper left')
    plt.show()


def lstm_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(LSTM(input_shape=(x_train.shape[1], x_train.shape[2]), units=100))
    model.add(Dropout(0.1))
    model.add(Dense(activation='softmax', units=3))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=60, validation_split=0.2)
    print(model.summary())
    result = model.evaluate(x_test, y_test)
    print(result)
    draw_history(history)
    model.save("fall_model.h5")


def deep_3D_cnn_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(Convolution3D(32, kernel_size=(9, 9, 9), activation='relu',
                            input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # model.add(BatchNormalization(center=True, scale=True))
    model.add(Dropout(0.1))
    # model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    # model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # # model.add(BatchNormalization(center=True, scale=True))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(activation='relu', units=128))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(activation='softmax', units=3))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
    print(model.summary())
    result = model.evaluate(x_test, y_test)
    print(result)
    draw_history(history)
    model.save("fall_3d_cnn_model.h5")


def deep_cnn_lstm_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(6, (3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], 1))))
    # model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.1)))

    model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    # model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.1)))

    # model.add(TimeDistributed(Conv2D(2, (2, 2), activation='relu'), input_shape=(x_train.shape[1], x_train.shape[
    # 2], x_train.shape[3], 1))) model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    # model.add(Dropout(0.1))
    model.add(LSTM(128))

    model.add(Dense(activation='softmax', units=3))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.005), metrics=['accuracy'])
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    history = model.fit(x_train, y_train, batch_size=64, epochs=120, validation_split=0.2)
    print(model.summary())
    result = model.evaluate(x_test, y_test)
    print(result)
    draw_history(history)
    model.save("fall_2d_cnn_model.h5")
