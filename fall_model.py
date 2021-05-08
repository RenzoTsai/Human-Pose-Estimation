from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Convolution3D, Conv3D, MaxPooling3D, BatchNormalization, Flatten, Conv2D, \
    MaxPooling2D, TimeDistributed, concatenate
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt


def baseline_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(Dense(input_dim=x_train.shape[1], activation='sigmoid', units=128))
    model.add(Dense(activation='relu', units=128))
    model.add(Dropout(0.5))
    model.add(Dense(activation='relu', units=128))
    model.add(Dropout(0.5))
    model.add(Dense(activation='softmax', units=3))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.005), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=60, validation_split=0.2)
    print(model.summary())
    result = model.evaluate(x_test, y_test)
    print(result)
    draw_history(history)
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
    model.add(LSTM(input_shape=(x_train.shape[1], x_train.shape[2]), units=128))
    model.add(Dropout(0.5))
    model.add(Dense(activation='softmax', units=3))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.005), metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2)
    print(model.summary())
    result = model.evaluate(x_test, y_test)
    print(result)
    draw_history(history)
    model.save("fall_model.h5")


def depth_3D_cnn_model(x_train=None, y_train=None, x_test=None, y_test=None):
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
    model.save("fall_3d_cnn_model.h5")


def depth_cnn_lstm_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(TimeDistributed(
        Conv2D(6, (5, 5), activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], 1))))
    # model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.5)))

    # model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    # # model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    # model.add(TimeDistributed(Dropout(0.1)))

    # model.add(TimeDistributed(Conv2D(2, (2, 2), activation='relu'), input_shape=(x_train.shape[1], x_train.shape[
    # 2], x_train.shape[3], 1))) model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    # model.add(Dropout(0.5))
    model.add(LSTM(100))

    model.add(Dense(32, activation='tanh'))

    model.add(Dense(activation='softmax', units=3))
    learning_rate = 0.0005
    nb_epoch = 100
    sgd = SGD(lr=learning_rate, decay=learning_rate / nb_epoch)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    history = model.fit(x_train, y_train, batch_size=32, epochs=nb_epoch, validation_split=0.2)
    print(model.summary())
    result = model.evaluate(x_test, y_test)
    print(result)
    draw_history(history)
    model.save("fall_2d_cnn_model.h5")


def get_point_model(inp):
    model = Sequential()
    model.add(LSTM(input_shape=(inp.shape[1], inp.shape[2]), units=128))
    model.add(Dropout(0.5))
    return model


def get_img_model(inp):
    inp = Input(shape=(inp.shape[1], inp.shape[2], inp.shape[3], 1))
    x = TimeDistributed(
        Conv2D(6, (5, 5), activation='relu', input_shape=(inp.shape[1], inp.shape[2], inp.shape[3], 1)))(inp)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Dropout(0.5))(x)

    # x = TimeDistributed(Conv2D(16, (3, 3)))(x)
    # x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    # x = TimeDistributed(Dropout(0.1))(x)
    # model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    # model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    # model.add(TimeDistributed(Dropout(0.1)))

    x = (TimeDistributed(Flatten()))(x)
    x = LSTM(100)(x)
    # x = Dropout(0.5)(x)
    x = Dense(32, activation='tanh')(x)
    x = Dropout(0.5)(x)
    return Model(inp, x)


def combined_model(x_img_train, x_point_train, y_train, x_img_test, x_point_test, y_test):
    img_model = get_img_model(x_img_train)

    point_model = get_point_model(x_point_train)

    combinedInput = concatenate([img_model.output, point_model.output])
    x = Dense(activation='relu', units=64)(combinedInput)
    x = Dense(activation='softmax', units=3)(x)
    model = Model(inputs=[img_model.input, point_model.input], outputs=x)
    learning_rate = 0.001
    nb_epoch = 250
    sgd = SGD(lr=learning_rate, decay=learning_rate / nb_epoch)
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learning_rate / nb_epoch, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())
    history = model.fit([x_img_train, x_point_train], y_train, batch_size=32, epochs=nb_epoch, validation_split=0.2)
    result = model.evaluate([x_img_test, x_point_test], y_test)
    print(result)
    draw_history(history)
    model.save("fall_combined_model.h5")
