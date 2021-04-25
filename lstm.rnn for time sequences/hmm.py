import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, GRU, LSTM
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from sklearn.utils import check_array, check_random_state
from hmmlearn import hmm

#改库函数
def sample(self, st, n_samples=1, random_state=None):
    """Generate random samples from the model.
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    random_state : RandomState or an int seed
        A random number generator instance. If ``None``, the object's
        ``random_state`` is used.
    Returns
    -------
    X : array, shape (n_samples, n_features)
        Feature matrix.
    state_sequence : array, shape (n_samples, )
        State sequence produced by the model.
    """
    # _utils.check_is_fitted(self, "startprob_")
    # self._check()

    if random_state is None:
        random_state = self.random_state
    random_state = check_random_state(random_state)

    # startprob_cdf = np.cumsum(self.startprob_)
    transmat_cdf = np.cumsum(self.transmat_, axis=1)

    # currstate = (startprob_cdf > random_state.rand()).argmax()
    currstate = st
    state_sequence = [currstate]
    X = [self._generate_sample_from_state(
        currstate, random_state=random_state)]
    #print(X)
    for t in range(n_samples - 1):
        currstate = (transmat_cdf[currstate] > random_state.rand()) \
            .argmax()#这里不断迭代就行了
        state_sequence.append(currstate)#没用得
        inter=self._generate_sample_from_state(
            currstate, random_state=random_state)
        #print(inter)
        X.append(inter)

    return np.atleast_2d(X), np.array(state_sequence, dtype=int)


maotai = pd.read_csv('./SH600519.csv')  # 读取股票文件

#生成训练集，测试集，2126：300
training_set = maotai.iloc[0:2426 - 300, 2:3].values  # 前(2426-300=2126)天的开盘价作为训练集,表格从0开始计数，2:3 是提取[2:3)列，前闭后开,故提取出C列开盘价
test_set = maotai.iloc[2426 - 300:, 2:3].values  # 后300天的开盘价作为测试集

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
test_set_scaled = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化

#训练
model = hmm.GaussianHMM(n_components=6, covariance_type="full", n_iter=100000)
model.fit(training_set_scaled)
model.n_features = 1

#对训练集解码得到最后时刻的隐状态

hidden_state = model.decode(training_set_scaled)[-1][-1]#这个记得是因为还是一个里面很杂得数组
print(hidden_state)


# 末状态输入模型进行预测，不需要滑动窗口，hmm牛逼
predicted_stock_price,hidstate=sample(model, hidden_state, 300)

# 对真实数据还原---从（0，1）反归一化到原始范围
#predicted_stock_price = np.array(predicted_stock_price)
#predicted_stock_price = predicted_stock_price[:, -1, 0]
predicted_stock_price=sc.inverse_transform(predicted_stock_price)
real_stock_price = test_set
# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MaoTai Stock Price')
plt.legend()
plt.show()

#误差计算
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, real_stock_price)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)

'''
# 训练集上效果
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)
print('shape:' + str(training_set_scaled.shape))
x_train = []
y_train = []
for i in range(10, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 10:i, 0])
    y_train.append(training_set_scaled[i - 9:i + 1, 0])
# 对训练集进行打乱
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 10, 1))
y_train = np.reshape(y_train, (y_train.shape[0], 10, 1))
predicted_stock_price = model.predict(x_train)
predicted_stock_price = np.reshape(predicted_stock_price, (predicted_stock_price.shape[0], 10, 1))
predicted_stock_price = predicted_stock_price[:, -1, 0]
# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_stock_price = np.reshape(predicted_stock_price, (predicted_stock_price.shape[0], 1))
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# 对真实数据还原---从（0，1）反归一化到原始范围
real_stock_price = sc.inverse_transform(training_set_scaled[10:])
# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='MaoTai Stock Price t')
plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price t')
plt.title('MaoTai Stock Price Prediction t')
plt.xlabel('Time t')
plt.ylabel('MaoTai Stock Price t')
plt.legend()
plt.show()

#训练集误差计算
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, real_stock_price)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)
'''