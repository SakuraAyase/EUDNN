import numpy as np
import copy
from sklearn import svm

def svm_predict(train_data, train_label, test_data, test_label, used_weight):
    model = svm.SVC()
    print('v')
    model.fit(train_data,train_label)
    print('v')
    z = test_data.dot(used_weight)
    a = relu(z)
    result = a.dot(used_weight.T)
    label = model.predict(result)
    acc = 0
    for i in range(len(label)):
        if(label[i] == test_label[i]):
            acc+=1
    return acc/len(label)


def relu(x):
    a = copy.deepcopy(x)
    print(a.shape)
    for i in range(len(a)):
        for j in range(len(a[0])):
            if(a[i,j]<0):
                a[i,j] = 0
    return a

def bin2int(a):
    result = 0
    for i in range(len(a)):
        result = result * 2
        result = result + a[i]
    return result


def nul(x):
    print(x.shape)
    index = 0
    for i in range(len(x[0])):
        if(x[0,i]!= float(0)):
            index = i
            break
    x = x.T
    n_lens = len(x)
    x_t = np.zeros(n_lens)


    x.shape = (784)
    a = [x]
    for i in range(0,index):
        temp = np.zeros(n_lens)
        for j in range(0,i+1):
            temp[j] = 1
        a.append(temp)

    for i in range(index+1,n_lens):
        temp = np.zeros(n_lens)
        for j in range(0,index+1):
            temp[j] = 1
        for j in range(index+1,i+1):
            temp[j] = 1
        a.append(temp)
    print(a[0].shape)
    a = np.array(a)



    b = np.zeros(a.shape)
    # 正交化
    for i in range(len(a)):
        b[i] = a[i]
        for j in range(0, i):
            b[i] -= np.dot(a[i], b[j]) / np.dot(b[j], b[j]) * b[j]
    # 归一化
    for i in range(len(b)):
        b[i] = b[i] / np.sqrt(np.dot(b[i], b[i]))
    print(b)
    return b


