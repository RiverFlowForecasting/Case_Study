# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def load_data(file_name):

    f = open(file_name)
    feature_data = []
    label_tmp = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(int(lines[-1]))
        feature_data.append(feature_tmp)
    f.close()


    m = len(label_tmp)
    n_class = len(set(label_tmp))

    label_data = np.mat(np.zeros((m, n_class)))
    for i in range(m):
        label_data[i, label_tmp[i]] = 1

    return np.mat(feature_data), label_data


class RBF_NN1():
    def __init__(self, hidden_nodes, input_data_trainX, input_data_trainY, testx, testy,ckpt_path):
        self.hidden_nodes = int(hidden_nodes)
        self.input_data_trainX = input_data_trainX
        self.input_data_trainY = input_data_trainY
        self.testx = testx
        self.testy = testy
        self.ckpt_path = ckpt_path



        n_input = (self.input_data_trainX).shape[1]
        n_output = (self.input_data_trainY).shape[1]
        X = tf.placeholder('float', [None, n_input], name='X')
        Y = tf.placeholder('float', [None, n_output], name='Y')
        c = tf.Variable(tf.random_normal([self.hidden_nodes, n_input]), name='c')
#            delta = tf.constant(value=0.0456,shape=[1,self.hidden_nodes],name="delta")
        delta = tf.Variable(tf.random_normal([1, self.hidden_nodes], stddev=1), name='delta')
        W = tf.Variable(tf.random_normal([self.hidden_nodes, n_output]), name='W')
        b = tf.Variable(tf.random_normal([1, n_output]), name='b')



        dist = tf.reduce_sum(tf.square(tf.subtract(tf.tile(X, [self.hidden_nodes, 1]), c)), 1)
        dist = tf.multiply(1.0, tf.transpose(dist))
        delta_2 = tf.square(delta)
        RBF_OUT = tf.exp(tf.multiply(-1.0, tf.divide(dist, tf.multiply(2.0, delta_2))))
        output_in = tf.matmul(RBF_OUT, W) + b
        y_pred = tf.nn.relu(output_in)
        cost = tf.reduce_mean(tf.pow(Y - y_pred, 2))
        train_op = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

        self.train_op = train_op
        self.X = X
        self.Y = Y
        self.cost = cost
        self.y_pred = y_pred

    def fit(self, trainX, trainY, epochs):
        trX = trainX
        trY = trainY
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            ##初始化所有参数
            tf.global_variables_initializer().run()
            print("fit begin!")
            for epoch in range(epochs):
                print("epoch:",epoch)
                for i in range(8000):
                    feed = {self.X: trX[i], self.Y: trY[i]}
                    sess.run(self.train_op, feed_dict=feed)
            print('Training complete!')
            saver.save(sess, save_path=self.ckpt_path)

    def predict(self, x, y):
        pred_trX = np.mat(np.zeros((len(x), y.shape[1])))
        correct_tr = 0.0
        saver = tf.train.Saver()
#        ckpt = tf.train.get_checkpoint_state('./model/train1.ckpt')
#        print(ckpt)
#        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_pa th +'.meta')
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver.restore(sess, self.ckpt_path)
            for i in range(len(x)):
                pred_tr = sess.run(self.y_pred, feed_dict={self.X: x[i]})
                pred_trX[i, :] = pred_tr
#            self.save_model('RBF_predict_results.txt', pred_trX)
        return pred_trX

    def save_model(self, file_name, weights):
        f_w = open(file_name, 'w')
        m, n = np.shape(weights)
        for i in range(m):
            w_tmp = []
            for j in range(n):
                w_tmp.append(str(weights[i, j]))
            f_w.write('\t'.join(w_tmp) + '\n')
        f_w.close()


def read_data(num_station=0):
    rol = 16
    # ------------------read train data...
    print("read train data...")
    x_train = np.loadtxt('trainx.txt')
    y_train = np.loadtxt('trainy.txt')
    x_train = x_train.reshape(-1, rol)
    y_train = y_train.reshape(-1, 1)
    x_tr = x_train.tolist()
    y_tr = y_train.tolist()
    i = 0
    i, x0_train, x1_train, x2_train, x3_train, y0_train, y1_train, y2_train, y3_train = 0, [], [], [], [], [], [], [], []
    while i < 24000:
        x0_train.append(x_tr[i])
        x1_train.append(x_tr[i + 1])
        x2_train.append(x_tr[i + 2])
        x3_train.append(x_tr[i + 3])
        # ---------------------------------
        y0_train.append(y_tr[i])
        y1_train.append(y_tr[i + 1])
        y2_train.append(y_tr[i + 2])
        y3_train.append(y_tr[i + 3])
        i += 4

    # read train data...
    print("read test data...")

    x_te, y_te = [], []
    if num_station == 0:
        np.load("test_river1001_x.npy",x_te)
        np.load("test_river1001_y.npy",y_te)
#        x_tr, y_tr = x0_train, y0_train
    if num_station == 1:
        np.load("test_river2002_x.npy",x_te)
        np.load("test_river2002_y.npy",y_te)
#        x_tr, y_tr = x1_train, y1_train
    if num_station == 2:
        np.load("test_river4001_x.npy",x_te)
        np.load("test_river4001_y.npy",y_te)
#        x_tr, y_tr = x2_train, y2_train
    if num_station == 3:
        np.load("test_river28012_x.npy",x_te)
        np.load("test_river28012_y.npy",y_te)
#        x_tr, y_tr = x3_train, y3_train

    return np.mat(x_tr), np.mat(y_tr), np.mat(x_te), np.mat(y_te)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true),axis=-1)*100    


if __name__ == '__main__':
    print('------------------------1.Load Data---------------------')
    train_x, train_y, test_x, test_y = read_data(num_station=3)
    print('------------------------2.parameter setting-------------')
    print("train_x shape", train_x.shape)
    print("train_y shape", train_y.shape)
    print("test_x  shape", test_x.shape)
    print("test_y  shape", test_y.shape)

    hidden_nodes = 100
    ckpt_path = "./model/train1-80.ckpt"   #hidden_nodes = 100
    input_data_trainX = train_x
    input_data_trainY = train_y
    rbf = RBF_NN1(hidden_nodes, input_data_trainX, input_data_trainY, test_x, test_y,ckpt_path)
#    rbf.fit(trainX=input_data_trainX, trainY=input_data_trainY,epochs=50)
#
    y_pred = rbf.predict(test_x, test_y)
    i = 0
    a, b, sub, sta = [], [], [], []
    while i < 270:
        a.append(y_pred[i][0] * 393.1)
        b.append(test_y[i][0] * 393.1)
        i += 1
    la = np.array(a).reshape(-1).tolist()
    lb = np.array(b).reshape(-1).tolist()
    print("the rmse is:", np.sqrt(mean_squared_error(la, lb)))
    print("the mae is:", mean_absolute_error(la, lb))
    print("the mape is：", mean_absolute_percentage_error(np.array(lb), np.array(la)))
#    np.savetxt("farm4.txt",lb)
#    np.savetxt("predict4.txt",la)
