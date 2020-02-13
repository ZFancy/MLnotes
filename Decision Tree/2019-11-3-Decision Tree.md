---
title: Decision Tree
tags: Machine_learning
show_subscribe: false
author: Zhu Jianing
---

### Decision Tree

------

简介分类决策树的基本概念，算法流程，划分属性的几种准则，剪枝处理及树的构建实现。

<!--more-->

#### 基本简介

Decision Tree 是基于树形结构进行决策的一种机器学习方法

通常可见，内部节点对应属性测试，叶节点对应于决策结果

```python
#决策树的基本算法流程-递归生成决策树
#训练集 D
#属性集 A
def Generate(D,A):
    生成结点 Node
    if D 样本全属于同一类别 C :
        将 Node 标记为 C 类别叶结点;return
    if A = empty or D 样本在 A 上取值相同 :
        将 Node 标记为叶结点，类别选择 D 中样本数最多的;return
    从 A 中选择最优划分属性 a
    for a 的每个取值 :
        生成 Node 的分支，获取样本子集
        if 样本子集为空 :
            将分支结点标记为叶结点，类别选择 D 中样本数最多的;return
        else
        	Generate(样本子集，A-a)
```



#### 特征选择（划分选择）

即选择最优划分属性，使分支结点包含样本尽可能归为同类，即高纯度。

##### 借助信息增益

信息熵(information entropy)，用以度量样本纯度的指标，值越小则样本纯度越高，定义为
$$
Ent(D)=-{\sum_{k=1}^{n}}p_klog_2p_k\\
$$
其中$p_k(k=1,2,3,...,n)$ 为第k类样本所占总集合的比例，即$\frac{|D_{k}|}{|D|}$

信息增益(information gain)，用以计算样本属性对集合划分的影响，通过单一离散属性的多个取值，计算各取值种类样本信息熵，以样本数越多影响越大的规则赋予权重，也即特征a对数据集D的经验条件熵，定义为
$$
Gain(D,a)=Ent(D)-\sum_{v=1}^V\frac{|D^v|}{|D|}Ent(D^v)
$$
ID3决策树算法即用信息增益为准则在属性集合A中划分属性，目标函数为
$$
\arg max Gain(D,a)\\a\in A
$$

##### 借助增益率

反思以上定义不难发现，信息增益对可取数目较多属性会有所倾向，C4.5决策树算法使用增益率(gain ratio)来划分属性，定义为
$$
Gain\_ratio(D,a)=\frac{Gain(D,a)}{-\sum_{v=1}^V\frac{|D^v|}{|D|}log_2\frac{|D^v|}{|D|}}
$$
C4.5算法使用了启发式：从候选划分属性中找到信息增益高于平均水平的属性，再从中找寻拥有最高增益率的属性。

##### 借助基尼指数

CART决策树算法使用基尼指数(Gini index)，直观理解为数据集中随机抽取两个样本，其类别不一致的概率，定义为
$$
Gini(D)=\sum_{k=1}^n\sum_{k'\neq k}p_kp_{k'}=1-\sum_{k=1}^np_{k}^2
$$
而属性a的基尼指数定义为
$$
Gain\_index(D,a)=\sum_{v=1}^V\frac{|D^v|}{|D|}Gini(D^v)
$$
在属性集合A中，选取划分后基尼指数最小的属性作为最优划分属性，目标函数为
$$
\arg min Gini\_index(D,a)\\
a\in A
$$

#### 剪枝处理

预剪枝：基于信息增益准则，在每一次选择划分属性进行划分后测试验证集精度，来判断是否要进行此次划分。

后剪枝：在训练生成一棵完整的决策树后通过剪枝及验证集精度判断，决定部分划分结点的保留或去除

#### 多变量决策树

单变量决策树对特征空间的划分具有划分界面平行于特征坐标的特点，对于一些复杂的分类任务，可能会使划分属性过多引起泛化性差，过拟合的影响。

多变量决策树的内部结点代表的往往是综合多个属性而来的划分条件，而非单属性，试图建立一个合适的线性分类器。
