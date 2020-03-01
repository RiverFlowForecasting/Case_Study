# coding=utf-8
from keras import backend as K
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, Dense, Reshape
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.optimize import rosen, differential_evolution


def mean_absolute_percentage_error(y_true, y_pred):
    y_mean = np.mean(y_true)
    return np.mean(np.abs((y_true - y_pred) / y_mean), axis=-1) * 100


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# fix random seed for reproducibility
seed = 7
num_rivers, num_history = 4, 4
np.random.seed(seed)

# load pima indians dataset
trainx = 'trainx.txt'
trainy = 'trainy.txt'
x_train = np.loadtxt(trainx)
y_train = np.loadtxt(trainy)
x = np.array(x_train)
y = np.array(y_train)
x_train = x.reshape(-1, num_history, num_history, num_rivers)
y_train = y.reshape(-1, 4)

i, y0_train, y1_train, y2_train, y3_train = 0, [], [], [], []
while i < 6000:
    # ---------------------------------
    y0_train.append(y_train[i, 0])
    y1_train.append(y_train[i, 1])
    y2_train.append(y_train[i, 2])
    y3_train.append(y_train[i, 3])
    i += 1
print(len(y_train))
print(len(y0_train))

# =====================================================
# testx = 'testx.txt'
# testy = 'testy.txt'
# x_test = np.loadtxt(testx)
# y_test = np.loadtxt(testy)
# x = np.array(x_test)
# y = np.array(y_test)
# x_test = x.reshape(-1, num_history, num_history, num_rivers)
# y_test = y.reshape(-1, 4)
x_test = np.load("gcnn_test_x.npy")



river_index=0
if river_index==0:
    print("Model for F1")
    lr = 0.0005
    inputs = Input(shape=(num_history, num_history, num_rivers))
    x1 = Conv2D(4, kernel_size=(2,2), input_shape=( num_history, num_history, num_rivers), padding="same", activation='relu')(inputs)
    x2 = Conv2D(4, kernel_size=(2,2), padding="valid", activation='relu')(x1)
    x3 = Conv2D(4, kernel_size=(2,2), padding="valid", activation='relu')(x2)
    m1 = MaxPooling2D(pool_size=(2,2), padding="valid", strides=1)(x3)
    x10 =Reshape((1,4))(m1)
    x12 = Dense(1)(x10)
    model = Model(inputs=inputs, outputs=x12)
    fname_param = os.path.join('F1.h5')
    true_value_station, pred_value_station = "true_value_station_G-CNN_1001.txt","pred_value_station_G-CNN_1001.txt"
    figure_name = "Farm1_CNN_4-1_MSE.pdf"
    adam = Adam(lr=lr)#------------
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    # print(model.summary())
    early_stopping = EarlyStopping(monitor="mean_squared_error", patience=15, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor="mean_squared_error", verbose=2, save_best_only=True, mode='min')#-------
    #Fit the model
    y=np.array(y0_train).reshape(-1,1,1)
    # yy_test = y0_test
    yy_test = np.load("y0_test.npy")
    # model.fit(x_train, y, epochs=500, batch_size=64, verbose=2,callbacks=[model_checkpoint])
    # model.save_weights(fname_param, overwrite=True)

    model.load_weights(fname_param)
    # calculate predictions
    pred = model.predict(x_test)
    print("prediction:", len(pred))
    pred = pred.reshape(-1, 1).tolist()
    a, b, sub, sta = [], [], [], []
    s, total = 0, 270

    while s < total:
        a.append(pred[s][0] * 393.1)
        b.append(yy_test[s] * 393.1)
        s += 1
    np.savetxt(true_value_station, b)
    np.savetxt(pred_value_station, a)
    print("the rmse is：", np.sqrt(mean_squared_error(a, b)))
    print("the mae is：", mean_absolute_error(a, b))
    print("the mape is：", mean_absolute_percentage_error(np.array(b), np.array(a)))

river_index=1
if river_index==1:
    print("Model for F2")
    lr = 0.00005
    inputs = Input(shape=(num_history, num_history, num_rivers))
    x1 = Conv2D(4, kernel_size=(2,2), input_shape=( num_history, num_history, num_rivers), padding="same", activation='relu')(inputs)
    x2 = Conv2D(4, kernel_size=(2,2), padding="same", activation='relu')(x1)
    x3 = Conv2D(4, kernel_size=(2,2), padding="same", activation='relu')(x2)
    m1 = MaxPooling2D(pool_size=(2,2), padding="valid", strides=1)(x3)
    x10 =Reshape((1,36))(m1)
    x12 = Dense(1)(x10)
    model = Model(inputs=inputs, outputs=x12)
    fname_param = os.path.join('F2.h5')
    figure_name = "Farm2_CNN_4-1_MSE.pdf"
    true_value_station, pred_value_station = "true_value_station_G-CNN_2002.txt","pred_value_station_G-CNN_2002.txt"
    adam = Adam(lr=lr)#------------
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    # print(model.summary())
    early_stopping = EarlyStopping(monitor="mean_squared_error", patience=15, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor="mean_squared_error", verbose=2, save_best_only=True, mode='min')#-------
    #Fit the model
    y=np.array(y1_train).reshape(-1,1,1)
    # yy_test = y1_test
    yy_test = np.load("y1_test.npy")
    # #model.fit(x_train, y, epochs=1000, batch_size=64, verbose=2,callbacks=[model_checkpoint])
    # #model.save_weights(fname_param, overwrite=True)


    model.load_weights(fname_param)
    # calculate predictions
    pred = model.predict(x_test)
    print("prediction:", len(pred))
    pred = pred.reshape(-1, 1).tolist()
    a, b, sub, sta = [], [], [], []
    s, total = 0, 270

    while s < total:
        a.append(pred[s][0] * 393.1)
        b.append(yy_test[s] * 393.1)
        s += 1
    np.savetxt(true_value_station, b)
    np.savetxt(pred_value_station, a)
    print("the rmse is：", np.sqrt(mean_squared_error(a, b)))
    print("the mae is：", mean_absolute_error(a, b))
    print("the mape is：", mean_absolute_percentage_error(np.array(b), np.array(a)))

river_index=2
if river_index==2:
    print("Model for F3")
    lr = 0.00005
    inputs = Input(shape=(num_history, num_history, num_rivers))
    x1 = Conv2D(4, kernel_size=(2,2), input_shape=( num_history, num_history, num_rivers), padding="same", activation='relu')(inputs)
    x2 = Conv2D(4, kernel_size=(2,2), padding="valid", activation='relu')(x1)
    x3 = Conv2D(5, kernel_size=(2,2), padding="valid", activation='relu')(x2)
    m1 = MaxPooling2D(pool_size=(2,2), padding="valid", strides=1)(x3)
    x10 =Reshape((1,5))(m1)
    x12 = Dense(1)(x10)
    model = Model(inputs=inputs, outputs=x12)
    fname_param = os.path.join('F3.h5')
    figure_name = "Farm3_CNN_4-1_MSE.pdf"
    true_value_station, pred_value_station = "true_value_station_G-CNN_4001.txt","pred_value_station_G-CNN_4001.txt"
    adam = Adam(lr=lr)#------------
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    # print(model.summary())
    early_stopping = EarlyStopping(monitor="mean_squared_error", patience=15, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor="mean_squared_error", verbose=2, save_best_only=True, mode='min')#-------
    #Fit the model
    y=np.array(y2_train).reshape(-1,1,1)
    # yy_test = y2_test
    # model.fit(x_train, y, epochs=1000, batch_size=64, verbose=2,callbacks=[model_checkpoint])
    # model.save_weights(fname_param, overwrite=True)

    yy_test = np.load("y2_test.npy")
    # #model.fit(x_train, y, epochs=1000, batch_size=64, verbose=2,callbacks=[model_checkpoint])
    # #model.save_weights(fname_param, overwrite=True)


    model.load_weights(fname_param)
    # calculate predictions
    pred = model.predict(x_test)
    print("prediction:", len(pred))
    pred = pred.reshape(-1, 1).tolist()
    a, b, sub, sta = [], [], [], []
    s, total = 0, 270

    while s < total:
        a.append(pred[s][0] * 393.1)
        b.append(yy_test[s] * 393.1)
        s += 1
    np.savetxt(true_value_station, b)
    np.savetxt(pred_value_station, a)
    print("the rmse is：", np.sqrt(mean_squared_error(a, b)))
    print("the mae is：", mean_absolute_error(a, b))
    print("the mape is：", mean_absolute_percentage_error(np.array(b), np.array(a)))

river_index=3
if river_index==3:
    print("Model for F4")
    lr = 0.00005
    inputs = Input(shape=(num_history, num_history, num_rivers))
    x1 = Conv2D(4, kernel_size=(2, 2), input_shape=(num_history, num_history, num_rivers), padding="same",
                activation='relu')(inputs)
    x2 = Conv2D(4, kernel_size=(2, 2), padding="same", activation='relu')(x1)
    x3 = Conv2D(4, kernel_size=(2, 2), padding="same", activation='relu')(x2)
    m1 = MaxPooling2D(pool_size=(2, 2), padding="valid", strides=1)(x3)
    x10 = Reshape((1, 36))(m1)
    x12 = Dense(1)(x10)
    model = Model(inputs=inputs, outputs=x12)
    fname_param = os.path.join('F4.h5')
    figure_name = "Farm4_CNN_4-1_MSE.pdf"
    true_value_station, pred_value_station = "true_value_station_G-CNN_28012.txt", "pred_value_station_G-CNN_28012.txt"
    adam = Adam(lr=lr)  # ------------
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    # print(model.summary())
    early_stopping = EarlyStopping(monitor="mean_squared_error", patience=15, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor="mean_squared_error", verbose=2, save_best_only=True, mode='min')  # -------
    # Fit the model
    y = np.array(y3_train).reshape(-1, 1, 1)
    # yy_test = y3_test
    # model.fit(x_train, y, epochs=500, batch_size=64, verbose=2,callbacks=[model_checkpoint])
    # model.save_weights(fname_param, overwrite=True)
    yy_test = np.load("y3_test.npy")
    # #model.fit(x_train, y, epochs=1000, batch_size=64, verbose=2,callbacks=[model_checkpoint])
    # #model.save_weights(fname_param, overwrite=True)


    model.load_weights(fname_param)
    # calculate predictions
    pred = model.predict(x_test)
    print("prediction:", len(pred))
    pred = pred.reshape(-1, 1).tolist()
    a, b, sub, sta = [], [], [], []
    s, total = 0, 270

    while s < total:
        a.append(pred[s][0] * 393.1)
        b.append(yy_test[s] * 393.1)
        s += 1
    np.savetxt(true_value_station, b)
    np.savetxt(pred_value_station, a)
    print("the rmse is：", np.sqrt(mean_squared_error(a, b)))
    print("the mae is：", mean_absolute_error(a, b))
    print("the mape is：", mean_absolute_percentage_error(np.array(b), np.array(a)))

