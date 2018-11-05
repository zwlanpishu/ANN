import numpy as np 
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义要使用的活化函数
def tanh(Z) : 
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    cache = Z

    assert(A.shape == Z.shape)    
    return A, cache

# 定义要使用的活化函数的求导
def tanh_backward(d_A, cache_activation) :
    Z = cache_activation
    s = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    d_Z = d_A * (1 - s ** 2)

    assert(d_Z.shape == Z.shape)
    return d_Z 

# 参数初始化函数
def initialize_parameters(layer_dims) : 

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)                   # 这里的L是神经网络包含输入层的总层数
    
    for layer in range(1, L) : 
        parameters["Weight" + str(layer)] = np.random.randn(layer_dims[layer], layer_dims[layer - 1]) / np.sqrt(layer_dims[layer-1])        
        parameters["Bias" + str(layer)] = np.zeros((layer_dims[layer], 1))
        assert(parameters["Weight" + str(layer)].shape == (layer_dims[layer], layer_dims[layer - 1]))
        assert(parameters["Bias" + str(layer)].shape == (layer_dims[layer], 1))

    return parameters

# 初始化使用Adam算法需要的参数
def initialize_Adam(parameters) : 
    num_layers = len(parameters) // 2
    velocity = {}
    square = {}
    for i in range(1, num_layers + 1) : 
        velocity["dW" + str(i)] = np.zeros(parameters["Weight" + str(i)].shape)
        velocity["dB" + str(i)] = np.zeros(parameters["Bias" + str(i)].shape)
        square["dW" + str(i)]   = np.zeros(parameters["Weight" + str(i)].shape)
        square["dB" + str(i)]   = np.zeros(parameters["Bias" + str(i)].shape)
    return velocity, square

# 神经网络前向传播——线性计算部分
def linear_forward(Weight, Bias, A_prev) : 
    
    Z = np.dot(Weight, A_prev) + Bias
    cache = (Weight, Bias, A_prev)

    assert(Z.shape == (Weight.shape[0], A_prev.shape[1]))
    return Z, cache

# 神经网络前向传播——神经元节点
def linear_forward_node(Weight, Bias, A_prev, activation = "tanh") : 
    
    if activation == "tanh" : 
        Z, cache_linear = linear_forward(Weight, Bias, A_prev)
        A, cache_activation = tanh(Z)

    assert(A.shape == (Weight.shape[0], A_prev.shape[1]))
    
    cache = (cache_linear, cache_activation)
    return A, cache

# 神经网络前向传播
def nn_model_forward(X, parameters) : 
    caches = []
    A = X
    num_layer = len(parameters) // 2                  # 神经网络的真实层数，不计输入层

    for i in range(1, num_layer) : 
        A_prev = A
        A, cache = linear_forward_node(parameters["Weight" + str(i)], 
                                       parameters["Bias" + str(i)],
                                       A_prev,
                                       "tanh")
        caches.append(cache)
    
    A_last, cache = linear_forward_node(parameters["Weight" + str(num_layer)],
                                        parameters["Bias" + str(num_layer)],
                                        A,
                                        "tanh")
    caches.append(cache)
    assert(A_last.shape == (1, X.shape[1]))
    return A_last, caches

# 计算损失函数, 这里将损失值定义为预测值与实际值的MSE
def compute_cost(A_last, Y) : 
    assert(A_last.shape == Y.shape)
    num_train = A_last.shape[1]
    cost = (1 / num_train) * (1 / 2) * np.sum((A_last - Y) ** 2)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

# 加入L2正则化之后的损耗计算
def compute_cost_with_regularization(A_last, Y, parameters, lambd = 0.0001) : 
    assert(A_last.shape == Y.shape)
    num_train = A_last.shape[1]
    num_layer = len(parameters) // 2
    cost_mse = (1 / num_train) * (1 / 2) * np.sum((A_last - Y) ** 2)
    cost_mse = np.squeeze(cost_mse)
    cost_regulation = 0
    for i in range(1, num_layer + 1) : 
        cost_regulation += np.sum(np.square(parameters["Weight" + str(i)]))
    cost_regulation = lambd * cost_regulation / (2 * num_train)
    cost = cost_mse + cost_regulation
    return cost

# 定义反向传播——线性求导部分
def linear_backward(dZ, cache_linear) :
    
    # 从缓存中取出对应的L层参数
    Weight = cache_linear[0]
    Bias = cache_linear[1]
    A_pre = cache_linear[2]

    # 确定样本的个数
    num_train = A_pre.shape[1]
    d_Weight = (1 / num_train) * np.dot(dZ, A_pre.T)
    d_Bias = (1 / num_train) * np.sum(dZ, axis = 1, keepdims = True)
    d_A_pre = np.dot(Weight.T, dZ)

    # 维度判断
    assert(d_Weight.shape == Weight.shape)
    assert(d_Bias.shape == Bias.shape)
    assert(d_A_pre.shape == A_pre.shape)

    return d_A_pre, d_Weight, d_Bias

# 定义反向传播——包含正则过程的线性求导部分
def linear_backward_with_regulation(dZ, cache_linear, lambd = 0.0001) : 
     # 从缓存中取出对应的L层参数
    Weight = cache_linear[0]
    Bias = cache_linear[1]
    A_pre = cache_linear[2]

    # 确定样本的个数
    num_train = A_pre.shape[1]
    d_Weight = (1 / num_train) * np.dot(dZ, A_pre.T) + (lambd / num_train) * Weight
    d_Bias = (1 / num_train) * np.sum(dZ, axis = 1, keepdims = True)
    d_A_pre = np.dot(Weight.T, dZ)

    # 维度判断
    assert(d_Weight.shape == Weight.shape)
    assert(d_Bias.shape == Bias.shape)
    assert(d_A_pre.shape == A_pre.shape)
    return d_A_pre, d_Weight, d_Bias

# 定义反向传播——神经元节点求导
def linear_backward_node(d_A, cache, activation = "tanh") :
    cache_linear, cache_activation = cache

    if activation == "tanh" :
        dZ = tanh_backward(d_A, cache_activation)
        d_A_pre, d_Weight, d_Bias = linear_backward_with_regulation(dZ, cache_linear)

    return d_A_pre, d_Weight, d_Bias

# 神经网络反向传播
def nn_model_backward(A_last, Y, caches) : 
    assert(A_last.shape == Y.shape)

    grads = {}
    num_layer = len(caches)
    d_A_last = A_last - Y
    grads["d_A" + str(num_layer)], grads["d_Weight" + str(num_layer)], grads["d_Bias" + str(num_layer)] \
        = linear_backward_node(d_A_last, caches[num_layer - 1], "tanh")
    
    for i in reversed(range(1, num_layer)) : 
        grads["d_A" + str(i)], grads["d_Weight" + str(i)], grads["d_Bias" + str(i)] \
            = linear_backward_node(grads["d_A" + str(i + 1)], caches[i - 1], "tanh")
    return grads

# 神经网络参数更新
def parameters_update(parameters, grads, learning_rate) : 
    num_layer = len(parameters) // 2
    for i in range(1, num_layer + 1) : 
        parameters["Weight" + str(i)] = \
            parameters["Weight" + str(i)] - learning_rate * grads["d_Weight" + str(i)]
        parameters["Bias" + str(i)] = \
            parameters["Bias" + str(i)] - learning_rate * grads["d_Bias" + str(i)]
    
    return parameters

# 加入Adam优化的神经网络参数更新
def parameters_update_Adam(parameters, grads, learning_rate, velocity, square,
                           beta1, beta2, episilon, t) : 
    num_layer = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for i in range(1, num_layer + 1) : 
        velocity["dW" + str(i)] = beta1 * velocity["dW" + str(i)] + \
                                  (1 - beta1) * grads["d_Weight" + str(i)]
        velocity["dB" + str(i)] = beta1 * velocity["dB" + str(i)] + \
                                  (1 - beta1) * grads["d_Bias" + str(i)]
        square["dW" + str(i)]   = beta2 * square["dW" + str(i)] + \
                                  (1 - beta2) * (grads["d_Weight" + str(i)] ** 2)
        square["dB" + str(i)]   = beta2 * square["dB" + str(i)] + \
                                  (1 - beta2) * (grads["d_Bias" + str(i)] ** 2)

        v_corrected["dW" + str(i)] = velocity["dW" + str(i)] / (1 - beta1 ** t)
        v_corrected["dB" + str(i)] = velocity["dB" + str(i)] / (1 - beta1 ** t)
        s_corrected["dW" + str(i)] = square["dW" + str(i)] / (1 - beta2 ** t)
        s_corrected["dB" + str(i)] = square["dB" + str(i)] / (1 - beta2 ** t)

        parameters["Weight" + str(i)] = parameters["Weight" + str(i)] - learning_rate * (
                                        v_corrected["dW" + str(i)] / 
                                        (np.sqrt(s_corrected["dW" + str(i)]) + episilon))
        parameters["Bias" + str(i)]   = parameters["Bias" + str(i)] - learning_rate * (
                                        v_corrected["dB" + str(i)] / 
                                        (np.sqrt(s_corrected["dB" + str(i)]) + episilon))
    
    return parameters, velocity, square

# 神经网络模型
def neural_network_model(X, Y, layer_dims, learning_rate = 0.001, num_iteration = 10000, print_cost = True) : 
    costs = []
    parameters = initialize_parameters(layer_dims)
    num_train = X.shape[1]

    for i in range(0, num_iteration) : 
        
        # 每次都进行次序重排
        permutation = list(np.random.permutation(num_train))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        assert(shuffled_X.shape[1] == num_train)
        assert(shuffled_Y.shape[1] == num_train)

        A_last, caches = nn_model_forward(shuffled_X, parameters)
        cost = compute_cost(A_last, shuffled_Y)
        grads = nn_model_backward(A_last, shuffled_Y, caches)
        parameters = parameters_update(parameters, grads, learning_rate)
        
        # 加入学习率衰减
        if i > 4000 : 
            learning_rate = 0.3

        
        if print_cost and i % 100 == 0 : 
            print("Cost after iteration %i ：%f" %(i, cost))
            costs.append(cost)

    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return parameters

def neutral_network_Adam(X, Y, layer_dims, learning_rate = 0.001, 
                         beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, 
                         num_iteration = 10000, print_cost = True) : 
    costs = []
    parameters = initialize_parameters(layer_dims)
    velocity, square = initialize_Adam(parameters)
    num_train = X.shape[1]
    times = 0

    for i in range(0, num_iteration) : 
        
        # 每次都进行次序重排
        permutation = list(np.random.permutation(num_train))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        assert(shuffled_X.shape[1] == num_train)
        assert(shuffled_Y.shape[1] == num_train)

        A_last, caches = nn_model_forward(shuffled_X, parameters)
        cost = compute_cost_with_regularization(A_last, shuffled_Y, parameters)
        grads = nn_model_backward(A_last, shuffled_Y, caches)

        times = times + 1
        parameters, velocity, square = parameters_update_Adam(parameters, grads, learning_rate, velocity, square,
                                            beta1, beta2, epsilon, times)
        
        if print_cost and i % 100 == 0 : 
            print("Cost after iteration %i ：%f" %(i, cost))
            costs.append(cost)

    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    return parameters

# 进行预测
def predict(test_X, parameters) : 
    test_Y, _ = nn_model_forward(test_X, parameters)
    assert(test_Y.shape[1] == test_X.shape[1])
    return test_Y

def load_train_data() : 
    train_X = np.arange(0, 2 * math.pi + 0.1, (2 * math.pi) / 8)
    train_X = np.sort(train_X, None)
    train_Y = np.sin(train_X)
    #train_Y = np.abs(np.sin(train_X))
    return train_X, train_Y

def load_train_data2() : 
    train_X1 = np.arange(-10, 10 + 0.1, 2.001)
    train_X2 = np.arange(-10, 10 + 0.1, 2.001)
    train_X = np.zeros((2, 11 * 11))
    train_Y = np.zeros((1, 11 * 11))
    sample_index = 0
    for X1 in list(train_X1) : 
        for X2 in list(train_X2) : 
            train_X[0][sample_index] = X1
            train_X[1][sample_index] = X2
            train_Y[0][sample_index] = math.sin(X1) * math.sin(X2) / (X1 * X2)
            sample_index  = sample_index + 1
    return train_X, train_Y

def load_test_data() : 
    test_X = np.random.uniform(0, 2 * math.pi, 361)
    test_X = np.sort(test_X, None)
    return test_X

def load_test_data2() : 
    test_X1 = np.random.uniform(-10, 10, 21)
    test_X2 = np.random.uniform(-10, 10, 21)
    test_X = np.zeros((2, 21 * 21))
    test_Y = np.zeros((1, 21 * 21))
    sample_index = 0
    for X1 in list(test_X1) : 
        for X2 in list(test_X2) : 
            test_X[0][sample_index] = X1
            test_X[1][sample_index] = X2
            
            if X1 == 0 : 
                X1 = X1 + 0.01
            if X2 == 0 : 
                X2 = X2 + 0.01

            test_Y[0][sample_index] = math.sin(X1) * math.sin(X2) / (X1 * X2)
            sample_index  = sample_index + 1
    return test_X, test_Y


# 确定神经网络的层数及每层的节点数(前面两题作图)
"""
train_X, train_Y = load_train_data()
test_X = load_test_data()
layer_dims = [1, 3, 3, 1]
train_X = np.reshape(train_X, (1, -1))
train_Y = np.reshape(train_Y, (1, -1))
test_X = np.reshape(test_X, (1, -1))
#parameters = neural_network_model(train_X, train_Y, layer_dims, learning_rate = 0.6)
parameters = neutral_network_Adam(train_X, train_Y, layer_dims, learning_rate = 0.001)
test_Y = predict(test_X, parameters)
test_Y_real = np.sin(test_X)
#test_Y_real = np.abs(np.sin(test_X))
plt.figure()
plt.plot(np.squeeze(test_X), np.squeeze(test_Y_real))
plt.plot(np.squeeze(test_X), np.squeeze(test_Y))
plt.show()
"""

# 后面一题作图
train_X, train_Y = load_train_data2()
test_X, test_Y_real = load_test_data2()
test_Y_real = np.sort(test_Y_real)
layer_dims = [2, 3, 3, 1]
parameters = neutral_network_Adam(train_X, train_Y, layer_dims, learning_rate = 0.001)
test_Y = predict(test_X, parameters)
test_Y = np.sort(test_Y)
plt.figure()
plt.plot(np.squeeze(test_Y_real))
plt.plot(np.squeeze(test_Y))
plt.show()


