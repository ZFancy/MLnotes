---
title: K Nearest Neighbor
tags: Machine_learning
show_subscribe: false
author: Zhu Jianing
---

#### K Nearest Neighbor

------

简易版实现及测试

```python
import numpy as np 
import operator
import matplotlib
import matplotlib.pyplot as plt 
from os import listdir

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
#plt.xlabel("每年获取飞行常客里程数")
#plt.ylabel("视频游戏所消耗时间百分比")
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

handwritingtest()
```





