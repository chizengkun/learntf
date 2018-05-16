#coding :utf-8
from tensorflow.examples.tutorials.mnist import  input_data


mnist = input_data.read_data_sets('./data',one_hot=True)
#样本数
print("train data size:", mnist.train.num_examples)
#返回验证样本数
print("validation data size:", mnist.validation.num_examples)
#返回测试集test样本数
print("test data size:", mnist.test.num_examples)
#print(mnist.train.labels[0])
#print(mnist.train.images[0])
BATCH_SIZE = 200
xs, ys = mnist.train.next_batch( BATCH_SIZE)
print("xs shape:", xs.shape)
print("ys shape:", ys.shape)
