# coding=utf-8
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, Dense, Reshape
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from tensorflow.python.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.optimize import differential_evolution

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# fix random seed for reproducibility
seed = 7
num_rivers, num_history = 4, 4
np.random.seed(seed)

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

de_test = np.load('de_test.npy')
print(de_test.shape)
y_test = y.reshape(-1, 4)

delta = 0.00079656


def delta_mse(y_true, y_pred):
    return K.mean(((((y_true - y_pred) / delta) ** 2 + 1) ** 0.5 - 1) * delta ** 2, axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    y_mean = np.mean(y_true)
    return np.mean(np.abs((y_true - y_pred) / y_mean), axis=-1) * 100


model_index = 4
if model_index == 1:
    fname_param = "F1.h5"
    print("Model for 1001")
    lr = 0.0005
    inputs = Input(shape=(4, 4, 4))
    x1 = Conv2D(4, kernel_size=(2, 2), input_shape=(4, 4, 4), padding="same", activation='relu')(inputs)
    x2 = Conv2D(4, kernel_size=(2, 2), padding="same", activation='relu')(x1)
    x3 = Conv2D(4, kernel_size=(2, 2), padding="same", activation='relu')(x2)
    m1 = MaxPooling2D(pool_size=(2, 2), padding="valid", strides=1)(x3)
    x10 = Reshape((1, 36))(m1)
    x12 = Dense(1)(x10)
    model = Model(inputs=inputs, outputs=x12)
    adam = Adam(lr=lr)  # ------------
    model.compile(loss=delta_mse, optimizer=adam, metrics=[delta_mse])
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor=delta_mse, verbose=2, save_best_only=True, mode='min')
    # Fit the model
    y = np.array(y0_train).reshape(-1, 1, 1)
    yy_test = np.load("test_data_1001.npy")
    true_path = "true_value_1001.npy"
    pred_path = "pred_value_1001.npy"
    # model.fit(x_train, y, epochs=500, batch_size=64, verbose=2,callbacks=[model_checkpoint])
    # model.save_weights(fname_param, overwrite=True)

if model_index == 2:
    print("Model for 2002")
    fname_param = "F2.h5"
    lr = 0.0005
    inputs = Input(shape=(4, 4, 4))
    x1 = Conv2D(4, kernel_size=(2, 2), input_shape=(4, 4, 4), padding="same", activation='relu')(inputs)
    x2 = Conv2D(4, kernel_size=(2, 2), padding="same", activation='relu')(x1)
    x3 = Conv2D(4, kernel_size=(2, 2), padding="same", activation='relu')(x2)
    m1 = MaxPooling2D(pool_size=(2, 2), padding="valid", strides=1)(x3)
    x10 = Reshape((1, 36))(m1)
    x12 = Dense(1)(x10)
    model = Model(inputs=inputs, outputs=x12)
    adam = Adam(lr=lr)  # ------------
    model.compile(loss=delta_mse, optimizer=adam, metrics=[delta_mse])
    print(model.summary())
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor=delta_mse, verbose=2, save_best_only=True, mode='min')  # -------
    # Fit the model
    y = np.array(y1_train).reshape(-1, 1, 1)
    yy_test = np.load("test_data_2002.npy")
    true_path = "true_value_2002.npy"
    pred_path = "pred_value_2002.npy"
    # model.fit(x_train, y, epochs=500, batch_size=64, verbose=2,callbacks=[model_checkpoint])
    # model.save_weights(fname_param, overwrite=True)

if model_index == 3:
    fname_param = "F3.h5"
    print("Model for 4001")
    lr = 0.0005
    inputs = Input(shape=(4, 4, 4))
    x1 = Conv2D(4, kernel_size=(2, 2), input_shape=(4, 4, 4), padding="same", activation='relu')(inputs)
    x2 = Conv2D(4, kernel_size=(2, 2), padding="same", activation='relu')(x1)
    x3 = Conv2D(4, kernel_size=(2, 2), padding="same", activation='relu')(x2)
    m1 = MaxPooling2D(pool_size=(2, 2), padding="valid", strides=1)(x3)
    x10 = Reshape((1, 36))(m1)
    x12 = Dense(1)(x10)
    model = Model(inputs=inputs, outputs=x12)
    adam = Adam(lr=lr)  # ------------
    model.compile(loss=delta_mse, optimizer=adam, metrics=[delta_mse])
    print(model.summary())
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor=delta_mse, verbose=2, save_best_only=True, mode='min')  # -------
    # Fit the model
    y = np.array(y2_train).reshape(-1, 1, 1)
    yy_test = np.load("test_data_4001.npy")
    true_path = "true_value_4001.npy"
    pred_path = "pred_value_4001.npy"
    # model.fit(x_train, y, epochs=500, batch_size=64, verbose=2,callbacks=[model_checkpoint])
    # model.save_weights(fname_param, overwrite=True)

if model_index == 4:
    print("Model for 28012")
    fname_param = "F4.h5"
    lr = 0.0005
    inputs = Input(shape=(4, 4, 4))
    x1 = Conv2D(4, kernel_size=(2, 2), input_shape=(4, 4, 4), padding="same", activation='relu')(inputs)
    x2 = Conv2D(4, kernel_size=(2, 2), padding="same", activation='relu')(x1)
    x3 = Conv2D(4, kernel_size=(2, 2), padding="same", activation='relu')(x2)
    m1 = MaxPooling2D(pool_size=(2, 2), padding="valid", strides=1)(x3)
    x10 = Reshape((1, 36))(m1)
    x12 = Dense(1)(x10)
    model = Model(inputs=inputs, outputs=x12)
    adam = Adam(lr=lr)  # ------------
    model.compile(loss=delta_mse, optimizer=adam, metrics=[delta_mse])
    print(model.summary())
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor=delta_mse, verbose=2, save_best_only=True, mode='min')  # -------
    # Fit the model
    y = np.array(y3_train).reshape(-1, 1, 1)
    yy_test = np.load("test_data_28012.npy")
    true_path = "true_value_28012.npy"
    pred_path = "pred_value_28012.npy"
    # model.fit(x_train, y, epochs=500, batch_size=64, verbose=2,callbacks=[model_checkpoint])
    # model.save_weights(fname_param, overwrite=True)


def fit_ness(delta):
    global iteration

    def delta_mse_de(y_true, y_pred):
        return K.mean(((((y_true - y_pred) / delta) ** 2 + 1) ** 0.5 - 1) * delta ** 2, axis=-1)

    model.compile(loss=delta_mse_de, optimizer=adam)
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor=delta_mse_de, verbose=0, save_best_only=True, mode='min')
    model.fit(x_train, y, epochs=200, batch_size=64, verbose=0, callbacks=[model_checkpoint])
    prediction0_value0 = model.predict(x_test).reshape(-1, 1).tolist()
    prediction1_value1 = [i[0] for i in prediction0_value0]
    loss_value = mean_squared_error(prediction1_value1, yy_test)
    print("current loss and iter:", loss_value, iteration)
    iteration += 1
    return loss_value


print("search the best delta!")
# iteration = 0

# bounds =  [(0,0.001)]
# result = differential_evolution(fit_ness, bounds, maxiter=20, popsize=20)
# print(result.x,result.fun)


# F1.h5->1001,F2.h5->2002,F3.h5->4001,F4.h5->28012


model.load_weights(fname_param)
# calculate predictions
pred = model.predict(de_test)
pred = pred.reshape(-1, 1).tolist()
a, b = [], []
s, total = 0, 270

while s < total:
    a.append(pred[s][0] * 393.1)
    b.append(yy_test[s] * 393.1)
    s += 1
np.save(true_path,b)
np.save(pred_path,a)
print("the rmse is：", np.sqrt(mean_squared_error(b, a)))
print("the mae is：", mean_absolute_error(a, b))
print("the mape is：", mean_absolute_percentage_error(np.array(b), np.array(a)))
