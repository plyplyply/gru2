import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout,Dense,SimpleRNN
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error

maotai=pd.read_csv('./SH600519.csv')
training_set=maotai.iloc[0:2426-300,2:6].values
test_set=maotai.iloc[2426-300:,2:6]#最后一个维度是有多个元素在的

sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)
test_set=sc.transform(test_set)

x_train=[]
y_train=[]

x_test=[]
y_test=[]

for i in range(60,len(training_set_scaled)):#60到个数减1
    x_train.append(training_set_scaled[i-60:i,:])#0-59，没毛病
    y_train.append(training_set_scaled[i,:])

np.random.seed(7)
x_train,y_train=np.array(x_train),np.array(y_train)
permut=np.random.permutation(x_train.shape[0])#生成的是0-59的排列

x_train,y_train=x_train[permut],y_train[permut]



x_train=np.reshape(x_train,(x_train.shape[0],60,4))

#测试集还是做成了每个都是60的样子，其实rnn的参数应该是支持更长的时间步的计算的，
# #可以试一试，但是毕竟原本的意义就是60-》1，换成61-》1都不大好
for i in range(60,len(test_set)):
    x_test.append(test_set[i-60:i,:])
    y_test.append((test_set[i,:]))
x_test,y_test=np.array(x_test),np.array(y_test)
x_test=np.reshape(x_test,(x_test.shape[0],60,4))
model=tf.keras.Sequential([
    SimpleRNN( 80,return_sequences=True),
    Dropout(0.2),
    SimpleRNN(100),
    Dropout(0.2),
    Dense(4)
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss='mean_squared_error',metrics=['accuracy'])#0.01是学习率
checkpoint_save_path='./checkpoint3for4finaldemention/rnn_stock.ckpt'
if os.path.exists(checkpoint_save_path+'.index'):
    print('------------------load model------------------------')
    model.load_weights(checkpoint_save_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss'
)
history=model.fit(x_train,y_train,batch_size=128,epochs=500,validation_data=(x_test,y_test),validation_freq=1,callbacks=[cp_callback])
model.summary()

loss=history.history['loss']
val_loss=history.history['val_loss']
plt.plot(loss,label='Training loss')
plt.plot(val_loss,label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

predicted_stock_price=[]
x_test=x_test[0]
x_test=np.reshape(x_test,(1,60,1))
for i in range(240):
    print(x_test)
    a = model.predict(x_test)
    b = sc.inverse_transform(a)
    a = np.squeeze(a)
    x_test[0, 0:59, 0] = x_test[0, 1:60, 0]  # 后者其实是1-60
    x_test[0, 59, 0] = a
# 对预测数据还原---从（0，1）反归一化到原始范围


    #print(a)
    #print(x_test.shape)
    predicted_stock_price.append(b)


    i+=1
    #print(x_test.shape)
# 对真实数据还原---从（0，1）反归一化到原始范围
predicted_stock_price=np.array(predicted_stock_price)
predicted_stock_price=np.reshape(predicted_stock_price,(predicted_stock_price.shape[0],1))
real_stock_price = sc.inverse_transform(test_set[60:])
# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='MaoTai Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price')
plt.title('MaoTai Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MaoTai Stock Price')
plt.legend()
plt.show()



##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, real_stock_price)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)

sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
training_set_scaled = sc.fit_transform(training_set)
x_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
# 对训练集进行打乱
# 将训练集由list格式变为array格式
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
predicted_stock_price = model.predict(x_train)
# 对预测数据还原---从（0，1）反归一化到原始范围
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# 对真实数据还原---从（0，1）反归一化到原始范围
real_stock_price = sc.inverse_transform(training_set_scaled[60:])
# 画出真实数据和预测数据的对比曲线
plt.plot(real_stock_price, color='red', label='MaoTai Stock Price t')
plt.plot(predicted_stock_price, color='blue', label='Predicted MaoTai Stock Price t')
plt.title('MaoTai Stock Price Prediction t')
plt.xlabel('Time t')
plt.ylabel('MaoTai Stock Price t')
plt.legend()
plt.show()
##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predicted_stock_price, real_stock_price)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predicted_stock_price, real_stock_price)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)
