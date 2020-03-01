# encoding = utf-8
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import os
import math


def mean_absolute_percentage_error(y_true, y_pred):
    y_mean = np.mean(y_true)
    return np.mean(np.abs((y_true - y_pred) / y_mean), axis=-1) * 100


rol = 16
# ------------------read train data...
print("read train data...")

river_ID = 3
if river_ID == 0:
    # 16，16，16，1
    x_te, y_te = np.load("x0_test.npy"), np.load("y0_test.npy")
    x_mlp, ymlp = np.load("x0_train.npy"), np.load("y0_train.npy")
    weighs_path, true_value_station, pred_value_station = "F1.h5", "true_value_station1001.txt", "pred_value_station1001.txt"
if river_ID == 1:
    # 18，16，16，1
    x_te, y_te = np.load("x1_test.npy"), np.load("y1_test.npy")
    x_mlp, ymlp = np.load("x1_train.npy"), np.load("y1_train.npy")
    weighs_path, true_value_station, pred_value_station = "F2.h5", "true_value_station2002.txt", "pred_value_station2002.txt"
if river_ID == 2:
    # 20，16，16，1
    x_te, y_te = np.load("x2_test.npy"), np.load("y2_test.npy")
    x_mlp, ymlp = np.load("x2_train.npy"), np.load("y2_train.npy")
    weighs_path, true_value_station, pred_value_station = "F3.h5", "true_value_station4001.txt", "pred_value_station4001.txt"
if river_ID == 3:
    # 20，16，16，1
    x_te, y_te = np.load("x3_test.npy"), np.load("y3_test.npy")
    x_mlp, ymlp = np.load("x3_train.npy"), np.load("y3_train.npy")
    weighs_path, true_value_station, pred_value_station = "F4.h5", "true_value_station28012.txt", "pred_value_station28012.txt"

model = Sequential()
model.add(Dense(20, input_shape=(16,)))
model.add(Dense(16))
model.add(Dense(1))

adam = Adam(lr=0.0005)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
# print(model.summary())
early_stopping = EarlyStopping(monitor="mean_squared_error", patience=15, mode='min')
model_checkpoint = ModelCheckpoint(
    weighs_path, monitor="mean_squared_error", verbose=2, save_best_only=True, mode='min')

# model.fit(x3_train, y3_train, epochs=300, batch_size=64, verbose=2, callbacks=[model_checkpoint])
# model.save_weights("F4.h5", overwrite=True)

model.load_weights(weighs_path)
# calculate predictions
pred_data = model.predict(np.array(x_te)).reshape(-1)
real_data = np.array(y_te).reshape(-1)

s, total = 0, 270
a,b=[],[]
while s < total:
    a.append(pred_data[s] * 393.1)
    b.append(real_data[s] * 393.1)
    s += 1
np.save(pred_value_station,a)
np.save(true_value_station,b)
MAE = mean_absolute_error(b, a)
MSE = mean_squared_error(b, a)
print("the mse is:", np.sqrt(MSE))
print("the mae is:", MAE)
print("the mape is：", mean_absolute_percentage_error(np.array(b), np.array(a)))


