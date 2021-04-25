import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,GRU,LSTM
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

maotai = pd.read_csv('./SH600519.csv')  # 读取股票文件

training_set = maotai.iloc[0:2426 - 300, 2:3].values  # 前(2426-300=2126)天的开盘价作为训练集,表格从0开始计数，2:3 是提取[2:3)列，前闭后开,故提取出C列开盘价
test_set = maotai.iloc[2426 - 300:, 2:3].values  # 后300天的开盘价作为测试集

# 归一化
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
#print(training_set_scaled)
test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化
#print(test_set)
x_train = []
y_train = []

x_test = []
y_test = []

# 测试集：csv表格中前2426-300=2126天数据
# 利用for循环，遍历整个训练集，提取训练集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建2426-300-60=2066组数据。
for i in range(10, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 10:i, 0])
    y_train.append(training_set_scaled[i-9:i+1, 0])

# 对训练集进行打乱
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)

# 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
# 此处整个数据集送入，送入样本数为x_train.shape[0]即2066组数据；输入60个开盘价，预测出第61天的开盘价，循环核时间展开步数为60; 每个时间步送入的特征是某一天的开盘价，只有1个数据，故每个时间步输入特征个数为1
x_train = np.reshape(x_train, (x_train.shape[0], 10, 1))
y_train = np.reshape(y_train, (y_train.shape[0], 10, 1))

# 测试集：csv表格中后300天数据
# 利用for循环，遍历整个测试集，提取测试集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建300-60=240组数据。
for i in range(10, len(test_set)):
    x_test.append(test_set[i - 10:i, 0])
    y_test.append(test_set[i-9:i+1, 0])
# 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 10, 1))
y_test = np.reshape(y_test, (y_test.shape[0], 10, 1))

'''
#模型构建
model = tf.keras.Sequential([
    GRU(50, return_sequences=True),
    Dropout(0.25),
    GRU(50, return_sequences=True),
    Dropout(0.25),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 损失函数用均方误差
# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
'''
checkpoint_save_path = "./checkpointgru12/gru_stock12.ckpt"
model = tf.keras.models.load_model('my_modelgru12.h5')
'''
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model weights not the model-----------------')
    #merely load the weights.not all of the model.performance soal with the increasement of the epochs
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

history = model.fit(x_train, y_train, batch_size=512, epochs=12, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

#model.summary()

file = open('./weightsgru12.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

model.save('my_modelgru12.h5')
'''
# model_load


##################################### predict ###############################
# 测试集输入模型进行预测
predicted_stock_price=[]
print('xtest_original_version:-----------------------------------------------------------------')
#np.set_printoptions(threshold=np.inf)
#print(x_test)
x_test=x_test[0]
#print(x_test)
x_test=np.reshape(x_test,(1,10,1))
for i in range(290):
    print('x_test:')
    print(x_test)
    a = model.predict(x_test)#一步一步进行的预测
    a = np.reshape(a,(1,-1))#这一步是为了能inverse
    b = sc.inverse_transform(a)

    a=np.squeeze(a)
    a=a[-1]#这是因为输出了全部的10个，迪哥说好训练。。。
    x_test[0, 0:9, 0] = x_test[0, 1:10, 0]  # 后者其实是1-60
    x_test[0, 9, 0] = a
# 对预测数据还原---从（0，1）反归一化到原始范围

    print('a')
    print(a)
    #print(x_test.shape)
    print('b')
    print(b)
    print()
    predicted_stock_price.append(b[-1])#这句话其实跟squeeze了差不多
    i+=1
    #print(x_test.shape)

# 对真实数据还原---从（0，1）反归一化到原始范围，但是预测数据不需要还原了，已经在滑动窗口中还原过
predicted_stock_price=np.array(predicted_stock_price)
predicted_stock_price=np.reshape(predicted_stock_price,(predicted_stock_price.shape[0],10,1))
predicted_stock_price=predicted_stock_price[:,-1,0]#十个十个都是最后一个是预测值
real_stock_price = sc.inverse_transform(test_set[10:,:])#前十个被拿去预测了,依旧是时间轴，y的维度否则没办法画图

# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MaoTai Stock Price')
plt.legend()
plt.show()

# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, real_stock_price)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)


print(predicted_stock_price)
print(real_stock_price)

'''
#######################################evaluate#################################################
sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)
print('shape:'+str(training_set_scaled.shape))
x_train = []
y_train = []
for i in range(10, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 10:i, 0])
    y_train.append(training_set_scaled[i-9:i+1, 0])

# 对训练集进行打乱
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 10, 1))
y_train = np.reshape(y_train, (y_train.shape[0], 10, 1))

#大于1的m值预测了会全部拼接起来，看起来是这样
predicted_stock_price = model.predict(x_train)

#预测出来的结果还改了一下格式
predicted_stock_price=np.reshape(predicted_stock_price,(predicted_stock_price.shape[0],10,1))
predicted_stock_price=predicted_stock_price[:,-1,0]

# 对预测数据还原---从（0，1）反归一化到原始范围，否则怕sc不能正确使用
predicted_stock_price=np.reshape(predicted_stock_price,(predicted_stock_price.shape[0],1))
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

