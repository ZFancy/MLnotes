---
title: K Nearest Neighbor
tags: Machine_learning
show_subscribe: false
author: Zhu Jianing
---

K nearest neighbor (KNN) ： 一种简单的算法，常用于分类问题

<!--more-->

|   类别   | 算法 |      适用案例      |      求解方向      | 损失函数 | 实现     |
| :------: | :--: | :----------------: | :----------------: | :------: | -------- |
| 监督学习 | KNN  | 多类分类，回归问题 | 寻找合适的距离定义 |    -     | 距离计算 |

### KNN 的简介及理解

K nearest neighbor ，即 K-近邻算法。

简单来说，在分类问题中，该算法的过程是依据待处理实例在某一具体的特征空间内的“邻近”样本来确定该实例的类别。

算法模型的处理过程为：

- 确定实例的特征坐标  


$$
E = (x_i,y_i)\\
E\in T,\,\ T = \{(x_1,y_1),(x_2,y_2),...,(x_n,y_n)\}
$$


- 确定一种距离度量，依据距离度量找出 K 个邻近实例 


$$
E\rightarrow E_{k},\,\ \min\{dist\}
$$


- 确定一种分类决策的规则来判断实例的类别

不难发现，该算法并没有普遍意义上的“学习”过程。

从中可以简要的得出KNN 算法的三个要素是：

1. K 值的选择；
2. 特征空间的距离度量；
3. 分类决策的规则；

__K 值的选择 __

如何选择合适的 K 值很大程度上取决于数据集的特征，和建模时的domain knowledge，考虑到部分特征空间的划分会受到噪声的影响， K 值不能取太小以防止过拟合的发生，当然值太大又会使误差变大，造成欠拟合的情况出现。实际情况中也可以通过交叉检验选取最优的 K 值。

__特征空间的距离度量__

特征空间中实例的距离常常用来反映实例间的相似程度。

这里介绍几种常用的距离定义：

设特征空间 $X$ 为 $n $ 维实数向量空间 $R^n$，$x_i,x_j\in X$  

_Minkowski distance_ （闵可夫斯基距离）


$$
D_M(x_i,x_j)=(\sum_{l=1}^n|x_{i}^{(l)}-x_{j}^{(l)}|^p)^{\frac{1}{p}}
$$


_Euclidean distance_ （欧式距离） 


$$
D_E(x_i,x_j)=(\sum_{l=1}^n|x_{i}^{(l)}-x_{j}^{(l)}|^2)^{\frac{1}{2}}
$$


_Manhattan distance_ （曼哈顿距离）


$$
D_{m}(x_i,x_j)=\sum_{l=1}^n|x_{i}^{(l)}-x_{j}^{(l)}|
$$


其实后面两种都是 Minkowski distance 的特殊情况，而当 $p\rightarrow \infty$ 时  


$$
D_{\infty}(x_i,x_j)=\max_{l}|x_{i}^{(l)}-x_{j}^{(l)}|
$$


其实还有很多种距离定义，对待具体的数据集或问题得具体分析。

__分类决策的规则__

一种简单的想法在 KNN 做出分类决策时是这样的，对选出的实例种类进行识别，坚持少数服从多数的规则，按多数表决给待处理实例打上标签。

当然实际问题也可将最邻近实例间的距离考虑到分类决策中来，按照距离远近来进行表决。

### KNN距离分类的简单python实现

```python
import numpy as np 
import operator
import matplotlib
import matplotlib.pyplot as plt 
from os import listdir

# Main KNN function
def KNN_classify(inx,dataset,labels,k):
    datasetsize=dataset.shape[0]
    diffmat=np.tile(inx,(datasetsize,1))-dataset
    sqdiffmat=diffmat**2
    sqdistances=sqdiffmat.sum(axis=1)
    distances=sqdistances**0.5
    sorteddistindicies=distances.argsort()
    classcount={}
    for i in range(k):
        votelabel=labels[sorteddistindicies[i]]
        classcount[votelabel]=classcount.get(votelabel,0)+1
    sortedclasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]


# Above is the main function of KNN
# Following are some test for KNN classifier 

def loaddataset(filename):
    loadfile=open(filename)
    num=len(loadfile.readlines())
    filemat=np.zeros((num,3))
    classlabel=[]
    loadfile=open(filename)
    index=0
    for i in loadfile.readlines():
        i=i.strip()
        exi=i.split('\t')
        filemat[index,:]=exi[0:3]
        classlabel.append(int(exi[-1]))
        index+=1
    return filemat,classlabel

def normalize(dataset):
    minval=dataset.min(0);maxval=dataset.max(0)
    ranges=maxval-minval
    normmat=np.zeros(dataset.shape)
    line=dataset.shape[0]
    normdataset=dataset-np.tile(minval,(line,1))
    normdataset=normdataset/np.tile(ranges,(line,1))
    return normdataset,ranges,minval

def datingtest():
    datingmat,datinglabel=loaddataset('datingTestSet2.txt')
    normmat,ranges,minval=normalize(datingmat)
    line=normmat.shape[0]
    numtest=int(line*0.20)
    errorcount=0.0
    for i in range(numtest):
        result=KNN_classify(normmat[i,:],normmat[numtest:line,:],datinglabel[numtest:line],3)
        print("the KNN_classify came back with: %d, the real answer is: %d" % (result,datinglabel[i]))
        if(result!=datinglabel[i]):errorcount+=1.0
    print("the total error rate is: %.3f" % (errorcount/float(numtest)))
    print(errorcount)

datingtest()

datingmat,datinglabel=loaddataset('datingTestSet2.txt')
fig=plt.figure()
ax=fig.add_subplot(212)
ax.scatter(datingmat[:,0],datingmat[:,1],15.0*np.array(datinglabel),15.0*np.array(datinglabel))
plt.show()

def loadimg(filename):
    returnmat=np.zeros((1,1024))
    imgfile=open(filename)
    for i in range(32):
        line=imgfile.readline()
        for j in range(32):
            returnmat[0,32*i+j]=int(line[j])
    return returnmat

def handwritingtest():
    labels=[]
    traininglist=listdir('trainingDigits')
    m=len(traininglist)
    trainingmat=np.zeros((m,1024))
    for i in range(m):
        filename=traininglist[i]
        filestr=filename.split('.')[0]
        classnumstr=int(filestr.split('_')[0])
        labels.append(classnumstr)
        trainingmat[i,:]=loadimg('trainingDigits/%s'%filename)
    testfilelist=listdir('testDigits')
    errorcount=0.0
    mtest=len(testfilelist)
    for i in range(mtest):
        filename=testfilelist[i]
        filestr=filename.split('.')[0]
        classnumstr=int(filestr.split('_')[0])
        vectortest=loadimg('testDigits/%s' % filename)
        results=KNN_classify(vectortest,trainingmat,labels,3)
        print("the classifier came back with: %d, the real answer is: %d" % (results,classnumstr))
        if(results!=classnumstr):errorcount+=1
    print("the total number of error is: %d" %errorcount)
    print("the total error rate is: %.3f" % (errorcount/float(mtest)))
```



