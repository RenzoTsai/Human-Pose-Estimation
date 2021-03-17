from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import SGD


def baseline_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(Dense(input_dim=x_train.shape[1], activation='sigmoid', units=100))
    model.add(Dense(activation='relu', units=100))
    model.add(Dropout(0.1))
    model.add(Dense(activation='relu', units=100))
    model.add(Dropout(0.1))
    model.add(Dense(activation='softmax', units=2))
    model.compile(loss='mse', optimizer=SGD(lr=0.05), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=60)
    result = model.evaluate(x_test, y_test)
    print(result)
    model.save("fall_baseline_model.h5")


def lstm_model(x_train=None, y_train=None, x_test=None, y_test=None):
    model = Sequential()
    model.add(LSTM(input_shape=(x_train.shape[1], x_train.shape[2]), units=100))
    model.add(Dropout(0.1))
    model.add(Dense(activation='softmax', units=3))
    model.compile(loss='mse', optimizer=SGD(lr=0.05), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=64, epochs=60)
    result = model.evaluate(x_test, y_test)
    print(result)
    model.save("fall_model.h5")
