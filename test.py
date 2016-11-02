#import keras
from keras.models import Sequential
from keras.layers import LSTM
import pandas as pd
import keras.layers.core as core
from sklearn.preprocessing import MinMaxScaler

def create_model(input_length):
    model = Sequential()
    model.add(LSTM(8, input_length=input_length, input_dim=1))
    model.add(core.Dense(8))
    model.add(core.Activation('tanh'))
    model.add(core.Dense(4))
    model.add(core.Activation('tanh'))
    model.add(core.Dense(1))
    model.add(core.Activation('linear'))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])
    print 'num_params = ' + str(model.count_params())
    return model


def main():
    # model = create_model(201)
    #
    # train_data = pd.read_csv('data/train_features.csv', header = None).values
    # train_labels = pd.read_csv('data/train_labels.csv', header = None).values
    # validation_data = pd.read_csv('data/val_features.csv', header=None).values
    # validation_labels = pd.read_csv('data/val_labels.csv', header = None).values
    #
    # features_scaler = MinMaxScaler()
    # labels_scaler = MinMaxScaler()
    # scaled_features = features_scaler.fit_transform(train_data.reshape(-1, 1)).reshape(train_data.shape[0], train_data.shape[1], 1)
    # scaled_labels = labels_scaler.fit_transform(train_labels.reshape(-1,1)).reshape(train_labels.shape)
    #
    # scaled_validation_features = features_scaler.transform(validation_data.reshape(-1, 1)).\
    #     reshape(validation_data.shape[0], validation_data.shape[1], 1)
    # scaled_validation_labels = labels_scaler.transform(validation_labels.reshape(-1, 1)).\
    #     reshape(validation_labels.shape)
    #
    # model.fit(x = scaled_features, y = scaled_labels, nb_epoch=100, batch_size=20, verbose=2,
    #           validation_data=(scaled_validation_features, scaled_validation_labels))
    model = Sequential()

    #     model.add(LSTM(hidden_dim, input_shape=(timesteps, data_dim)))
    #     model.add(Dropout(0.2))
    data_dim = 4
    hidden_dim = 32
    output_dim = 4
    batch_size = 9
    nb_epoch = 100
    timesteps = 12
    model.add(LSTM(hidden_dim, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension hidden_dim
    model.add(core.Dropout(0.2))

    #     model.add(LSTM(hidden_dim, return_sequences=True))  # returns a sequence of vectors of dimension hidden_dim
    #     model.add(Dropout(0.2))

    model.add(LSTM(hidden_dim))  # return a single vector of dimension hidden_dim
    model.add(core.Dropout(0.2))

    model.add(core.Dense(output_dim=output_dim, input_dim=hidden_dim))
    model.add(core.Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    print 'model.params = ' + str(model.count_params())
    return model

if __name__=="__main__":
    main()