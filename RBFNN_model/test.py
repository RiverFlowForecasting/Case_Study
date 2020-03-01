import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.mean(y_true)), axis=-1) * 100


print("ID 1001")
real = np.loadtxt("farm1.txt").reshape(-1).tolist()
pred = np.loadtxt("predict1.txt").reshape(-1).tolist()
print("the rmse is:", np.sqrt(mean_squared_error(real, pred)))
print("the mae is:", mean_absolute_error(real, pred))
print("the mape is：", mean_absolute_percentage_error(np.array(real), np.array(pred)))

print("ID 2002")
real = np.loadtxt("farm2.txt").reshape(-1).tolist()
pred = np.loadtxt("predict2.txt").reshape(-1).tolist()
print("the rmse is:", np.sqrt(mean_squared_error(real, pred)))
print("the mae is:", mean_absolute_error(real, pred))
print("the mape is：", mean_absolute_percentage_error(np.array(real), np.array(pred)))

print("ID 4001")
real = np.loadtxt("farm3.txt").reshape(-1).tolist()
pred = np.loadtxt("predict3.txt").reshape(-1).tolist()
print("the rmse is:", np.sqrt(mean_squared_error(real, pred)))
print("the mae is:", mean_absolute_error(real, pred))
print("the mape is：", mean_absolute_percentage_error(np.array(real), np.array(pred)))

print("ID 28012")
real = np.loadtxt("farm4.txt").reshape(-1).tolist()
pred = np.loadtxt("predict4.txt").reshape(-1).tolist()
print("the rmse is:", np.sqrt(mean_squared_error(real, pred)))
print("the mae is:", mean_absolute_error(real, pred))
print("the mape is：", mean_absolute_percentage_error(np.array(real), np.array(pred)))