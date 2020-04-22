# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf

# 下载并载入MNIST手写数字库（55000*28*28）
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

intput_x = tf.placeholder(tf.float32, [None, 28*28]) / 255
output_y = tf.placeholder(tf.int32, [None, 10]) # 输出：10个数字的标签
input_x_images = tf.reshape(intput_x, [-1,28,28,1]) # 改变形状之后的输入

# 从 Test(测试) 数据集里选取3000个手写数字的图片和对应标签
test_x = mnist.test.images[:3000] # 图片
test_y = mnist.test.labels[:3000] # 标签

# 构建CNN
# 第1层卷积
conv1 = tf.layers.conv2d(
		inputs = input_x_images, # 形状 [28,28,1]
		filters = 32, 			# 32个过滤器，输出的深度32
		kernel_size = [5,5], 	# 过滤器形状
		strides = 1, 			# 步幅
		padding = 'same', 		# same表示的大小不变，需要在外围补零2圈
		activation = tf.nn.relu # 激活函数是ReLU
		)
		# 形状[28,28,32]

# 第1层池化（亚采样）
pool1 = tf.layers.max_pooling2d(
		inputs = conv1, 	# [28,28,32]
		pool_size = [2,2], 	# 过滤器
		stride = 2 			# 步幅是2
		)
		# 形状[14,14,32]

# 第2层卷积
conv2 = tf.layers.conv2d(
		inputs = pool1, 		# 形状 [14,14,32]
		filters = 64, 			# 64个过滤器，输出的深度64
		kernel_size = [5,5], 	# 过滤器形状
		strides = 1, 			# 步幅
		padding = 'same', 		# same表示的大小不变，需要在外围补零2圈
		activation = tf.nn.relu # 激活函数是ReLU
		)
		# 形状[14,14,64]

# 第2层池化
pool2 = tf.layers.max_pooling2d(
		inputs = conv2, 	# [14,14,64]
		pool_size = [2,2], 	# 过滤器
		stride = 2 			# 步幅是2
		)
		# 形状[7,7,64]

# 平坦化(flat)
flat = tf.reshape(pool2, [-1,7*7*64]) # 形状[7*7*64, ]

# 1024 全连接层
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

# Dropout: 丢弃50%, rate=0.5
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

# 10个神经元的全连接层，不用激活函数来做非线性化
logits = tf.layers.dense(inputs=dropout,units=10) # 输出，形状[1,1,10]

# 计算误差(计算交叉熵，Softmax)
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

# 用Adam优化器来最小化误差，学习率0.001
train_op = tf.train.AdamOptimizer(learing_rate = 0.001).minimize()

# 精度，预测 与 实际标签 的匹配程度
# 返回 (accuracy, update_op), 两个局部变量
accuracy = tf.metrics.accuracy(
			labels = tf.argmax(output_y, axis=1),
			predictions = tf.argmax(logits, axis=1),)[1]

# 创建会话
sess = tf.Session()
# 初始化变量：全局和局部
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

for i in range(20000):
	batch = mnist.train.next_batch(50) # 从 Train 数据集里取下一50样本
	train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
	if i%100 ==0 :
		test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
		print("Step=%d, Train loss=%.4f, [Test accuracy=%.2f]") % (i, train_loss, test_accuracy)

# 测试：打印20个预测值和真实值的对
test_output = sess.run(logits, {input_x: test_x[:20]})
inferenced_y = np.argmax(test_output, 1)
print(inferenced_y, 'Inferenced numbers') # 推测的数字
print(np.argmax(test_y[:20], 1), 'Real numbers') # 真实的数字

sess.close()
