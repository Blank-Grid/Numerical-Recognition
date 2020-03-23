import numpy as np
import scipy.special
import matplotlib.pyplot

#神经网络
class neuralNetwork:

    #初始化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        #链接权重矩阵
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # 训练
    def train(self, inputs_list, targets_list):

        #针对样本计算输出
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        #误差反向传播，更新权重
        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass


    #查询
    def query(self, inputs_list):

        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784
output_nodes = 10
hidden_nodes = 200  #至少大于100

learning_rate = 0.1  #没有进行科学的评估，仅手动试了几个数值，发现0.1性能较好

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#训练数据
for record in training_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01  #缩放和移位MNIST数据,范围为0.01-1.00
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)


#测试网络
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# all_values = test_data_list[0].split(',')
# print(all_values[0])
#
# image_array = np.asfarray(all_values[1:]).reshape((28, 28))
# matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
#
# matplotlib.pyplot.show()


scorecard = []

epochs = 4  #4个世代的训练

for e in range(epochs):

    for record in test_data_list:

        #分离文本
        all_values = record.split(',')
        #记录第一个数字，即正确答案
        correct_label = int(all_values[0])

        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)

        #最大值的索引
        label = np.argmax(outputs)

        if(label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass

        pass


#准确率(性能)
scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)


