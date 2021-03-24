from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Convolution3D, Conv3D, MaxPooling3D, BatchNormalization, Flatten, Conv2D, \
    MaxPooling2D, TimeDistributed, concatenate
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
    model.fit(x_train, y_train, batch_size=32, epochs=50)
    result = model.evaluate(x_test, y_test)
    print(result)
    model.save("./models/my_trained_model/fall_baseline_model.h5")


def draw_history(history):
    plt.plot(history.history['loss'], color='r')
    plt.plot(history.history['val_loss'], color='g')
    plt.plot(history.history['accuracy'], color='b')
    plt.plot(history.history['val_accuracy'], color='k')
    plt.title('Model Acc and Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'test_loss', 'train_acc', 'test_acc'], loc='upper left')
    plt.savefig("acc_loss.png")
    plt.show()

def lstm_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(LSTM(input_shape=(x_train.shape[1], x_train.shape[2]), units=100))
    model.add(Dropout(0.1))
    model.add(Dense(activation='softmax', units=3))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2)
    print(model.summary())
    result = model.evaluate(x_test, y_test)
    print(result)
    draw_history(history)
    model.save("/models/my_trained_model/fall_model.h5")


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
    history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2)
    print(model.summary())
    result = model.evaluate(x_test, y_test)
    print(result)
    draw_history(history)
    model.save("./models/my_trained_modelfall_3d_cnn_model.h5")


def deep_cnn_lstm_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(TimeDistributed(
        Conv2D(6, (3, 3), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], 1))))
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
    history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.2)
    print(model.summary())
    result = model.evaluate(x_test, y_test)
    print(result)
    draw_history(history)
    model.save("./models/my_trained_model/fall_cnn_lstm_model.h5")


def get_point_model(inp):
    model = Sequential()
    model.add(LSTM(input_shape=(inp.shape[1], inp.shape[2]), units=100))
    model.add(Dropout(0.1))
    return model


def get_img_model(inp):
    inp = Input(shape=(inp.shape[1], inp.shape[2], inp.shape[3], 1))
    # x = Sequential(input)
    x = TimeDistributed(
        Conv2D(6, (3, 3), activation='relu', input_shape=(inp.shape[1], inp.shape[2], inp.shape[3], 1)))(inp)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Dropout(0.1))(x)

    x = TimeDistributed(Conv2D(16, (3, 3)))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Dropout(0.1))(x)

    # model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    # model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    # model.add(TimeDistributed(Dropout(0.1)))

    x = (TimeDistributed(Flatten()))(x)
    x = LSTM(128)(x)
    return Model(inp, x)


def combined_model(x_img_train, x_point_train, y_train, x_img_test, x_point_test, y_test):
    img_model = get_img_model(x_img_train)

    point_model = get_point_model(x_point_train)

    combinedInput = concatenate([img_model.output, point_model.output])
    x = Dense(activation='relu', units=32)(combinedInput)
    x = Dense(activation='softmax', units=3)(x)
    model = Model(inputs=[img_model.input, point_model.input], outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.005), metrics=['accuracy'])
    print(model.summary())
    history = model.fit([x_img_train, x_point_train], y_train, batch_size=32, epochs=50, validation_split=0.2)
    result = model.evaluate([x_img_test, x_point_test], y_test)
    print(result)
    draw_history(history)
    model.save("./models/my_trained_model/fall_combined_model.h5")
