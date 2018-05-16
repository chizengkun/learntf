#coding :utf-8
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#定义输入参数
x =tf.placeholder(tf.float32, shape=(None,2))
w1=tf.Variable(tf.random_normal([2,3],stddev=1, seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1, seed=1))

#定义前向传播
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#会话结算
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y in tf1 is :\n",sess.run(y, feed_dict={x:[[0.7,0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))