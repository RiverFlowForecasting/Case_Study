# coding = utf-8
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from scipy.optimize import differential_evolution
import math


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.mean(y_true)), axis=-1) * 100


rol = 16

# chang the river ID from  0 to 3
river_index = 0
if river_index == 0:
    x_te, y_te = np.load("x0_test.npy"), np.load("y0_test.npy")
    x_krr, y_krr = np.load("x0_train.npy"), np.load("y0_train.npy")
    true_value_station, pred_value_station = "true_value_station1001.npy", "pred_value_station1001.npy"

    krr = KernelRidge(kernel='rbf', alpha=0.0005, gamma=0.8)
    krr.fit(x_krr, y_krr)

    krr_pred = krr.predict(x_te).reshape(-1).tolist()
    real_data = np.array(y_te).reshape(-1).tolist()
    i=0
    a, b = [], []
    while i < 270:
        a.append(real_data[i] * 393.1)
        b.append(krr_pred[i] * 393.1)
        i += 1
    np.savetxt(true_value_station,a)
    np.savetxt(pred_value_station,b)
    print("the rmse is:", np.sqrt(mean_squared_error(a, b)))
    print("the mae is:", mean_absolute_error(a, b))
    print("the mape is：", mean_absolute_percentage_error(np.array(a), np.array(b)))

river_index = 1
if river_index == 1:
    x_te, y_te = np.load("x1_test.npy"), np.load("y1_test.npy")
    x_krr, y_krr = np.load("x1_train.npy"), np.load("y1_train.npy")
    true_value_station, pred_value_station = "true_value_station2002.npy", "pred_value_station2002.npy"

    krr = KernelRidge(kernel='rbf', alpha=0.0005, gamma=0.8)
    krr.fit(x_krr, y_krr)

    krr_pred = krr.predict(x_te).reshape(-1).tolist()
    real_data = np.array(y_te).reshape(-1).tolist()
    i=0
    a, b = [], []
    while i < 270:
        a.append(real_data[i] * 393.1)
        b.append(krr_pred[i] * 393.1)
        i += 1
    np.savetxt(true_value_station,a)
    np.savetxt(pred_value_station,b)
    print("the rmse is:", np.sqrt(mean_squared_error(a, b)))
    print("the mae is:", mean_absolute_error(a, b))
    print("the mape is：", mean_absolute_percentage_error(np.array(a), np.array(b)))

river_index = 2
if river_index == 2:
    x_te, y_te = np.load("x2_test.npy"), np.load("y2_test.npy")
    x_krr, y_krr = np.load("x2_train.npy"), np.load("y2_train.npy")
    true_value_station, pred_value_station = "true_value_station4001.npy", "pred_value_station4001.npy"

    krr = KernelRidge(kernel='rbf', alpha=0.0005, gamma=0.8)
    krr.fit(x_krr, y_krr)

    krr_pred = krr.predict(x_te).reshape(-1).tolist()
    real_data = np.array(y_te).reshape(-1).tolist()
    i=0
    a, b = [], []
    while i < 270:
        a.append(real_data[i] * 393.1)
        b.append(krr_pred[i] * 393.1)
        i += 1
    np.savetxt(true_value_station,a)
    np.savetxt(pred_value_station,b)
    print("the rmse is:", np.sqrt(mean_squared_error(a, b)))
    print("the mae is:", mean_absolute_error(a, b))
    print("the mape is：", mean_absolute_percentage_error(np.array(a), np.array(b)))

river_index = 3
if river_index == 3:
    x_te, y_te = np.load("x3_test.npy"), np.load("y3_test.npy")
    x_krr, y_krr = np.load("x3_train.npy"), np.load("y3_train.npy")
    true_value_station, pred_value_station = "true_value_station28012.npy", "pred_value_station28012.npy"

    krr = KernelRidge(kernel='rbf', alpha=0.0005, gamma=0.8)
    krr.fit(x_krr, y_krr)

    krr_pred = krr.predict(x_te).reshape(-1).tolist()
    real_data = np.array(y_te).reshape(-1).tolist()
    i=0
    a, b = [], []
    while i < 270:
        a.append(real_data[i] * 393.1)
        b.append(krr_pred[i] * 393.1)
        i += 1
    np.savetxt(true_value_station,a)
    np.savetxt(pred_value_station,b)
    print("the rmse is:", np.sqrt(mean_squared_error(a, b)))
    print("the mae is:", mean_absolute_error(a, b))
    print("the mape is：", mean_absolute_percentage_error(np.array(a), np.array(b)))