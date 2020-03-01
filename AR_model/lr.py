import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def mean_absolute_percentage_error(y_true, y_pred):
    y_mean = np.mean(y_true)
    return np.mean(np.abs((y_true - y_pred) / y_mean), axis=-1) * 100

river_index=0
if river_index==0:
    x_te, y_te = np.load("x0_test.npy"), np.load("y0_test.npy")
    x_mlp, ymlp = np.load("x0_train.npy"), np.load("y0_train.npy")

    print("ID 1001")
    lr1 = LinearRegression()
    lr1.fit(x_mlp, ymlp)
    pred_data = lr1.predict(np.array(x_te)).reshape(-1)
    real_data = np.array(y_te).reshape(-1)
    i=0
    a, b = [], []
    while i < 270:
        a.append(real_data[i] * 393.1)
        b.append(pred_data[i] * 393.1)
        i += 1
    np.savetxt("true_value_station_1001.npy",a)
    np.savetxt("pred_value_station_1001.npy",b)
    MAE = mean_absolute_error(b, a)
    MSE = mean_squared_error(b, a)
    print("the rmse is:", np.sqrt(MSE))
    print("the mae is:", MAE)
    print("the mape is：", mean_absolute_percentage_error(np.array(a), np.array(b)))

river_index=1
if river_index==1:
    x_te, y_te = np.load("x1_test.npy"), np.load("y1_test.npy")
    x_mlp, ymlp = np.load("x1_train.npy"), np.load("y1_train.npy")

    print("ID 2002")
    lr1 = LinearRegression()
    lr1.fit(x_mlp, ymlp)
    pred_data = lr1.predict(np.array(x_te)).reshape(-1)
    real_data = np.array(y_te).reshape(-1)
    i=0
    a, b = [], []
    while i < 270:
        a.append(real_data[i] * 393.1)
        b.append(pred_data[i] * 393.1)
        i += 1
    np.savetxt("true_value_station_2002.npy",a)
    np.savetxt("pred_value_station_2002.npy",b)
    MAE = mean_absolute_error(b, a)
    MSE = mean_squared_error(b, a)
    print("the rmse is:", np.sqrt(MSE))
    print("the mae is:", MAE)
    print("the mape is：", mean_absolute_percentage_error(np.array(a), np.array(b)))

river_index=2
if river_index==2:
    x_te, y_te = np.load("x2_test.npy"), np.load("y2_test.npy")
    x_mlp, ymlp = np.load("x2_train.npy"), np.load("y2_train.npy")

    print("ID 4001")
    lr1 = LinearRegression()
    lr1.fit(x_mlp, ymlp)
    pred_data = lr1.predict(np.array(x_te)).reshape(-1)
    real_data = np.array(y_te).reshape(-1)
    i=0
    a, b = [], []
    while i < 270:
        a.append(real_data[i] * 393.1)
        b.append(pred_data[i] * 393.1)
        i += 1
    np.savetxt("true_value_station_4001.npy",a)
    np.savetxt("pred_value_station_4001.npy",b)
    MAE = mean_absolute_error(b, a)
    MSE = mean_squared_error(b, a)
    print("the rmse is:", np.sqrt(MSE))
    print("the mae is:", MAE)
    print("the mape is：", mean_absolute_percentage_error(np.array(a), np.array(b)))

river_index=3
if river_index==3:
    x_te, y_te = np.load("x3_test.npy"), np.load("y3_test.npy")
    x_mlp, ymlp = np.load("x3_train.npy"), np.load("y3_train.npy")

    print("ID 2002")
    lr1 = LinearRegression()
    lr1.fit(x_mlp, ymlp)
    pred_data = lr1.predict(np.array(x_te)).reshape(-1)
    real_data = np.array(y_te).reshape(-1)
    i=0
    a, b = [], []
    while i < 270:
        a.append(real_data[i] * 393.1)
        b.append(pred_data[i] * 393.1)
        i += 1
    np.savetxt("true_value_station_28012.npy",a)
    np.savetxt("pred_value_station_28012.npy",b)
    MAE = mean_absolute_error(b, a)
    MSE = mean_squared_error(b, a)
    print("the rmse is:", np.sqrt(MSE))
    print("the mae is:", MAE)
    print("the mape is：", mean_absolute_percentage_error(np.array(a), np.array(b)))
