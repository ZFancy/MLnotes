---
title: Expectation Maximaization(EM) algorithm
tags: Machine_learning
show_subscribe: false
author: Zhu Jianing
---

EM算法：用于含有隐变量的概率模型参数的极大似然估计

该篇整理回顾期望极大算法及高斯混合模型，GEM算法的学习

<!--more-->

|   类别   | 算法 |     适用案例     |            求解方向            |   损失函数   | 实现 |
| :------: | :--: | :--------------: | :----------------------------: | :----------: | ---- |
| 监督学习 |  EM  | 含隐变量概率模型 | 极大似然估计，最大后验概率估计 | 对数似然损失 | 迭代 |

### 相关定义

模型参数：  


$$
\theta=(\pi,p,q)
$$


观测数据(记录数据)及未观测数据(记录事件发生前的隐含数据)：  


$$
Y=(Y_1,Y_2,...,Y_n)^T,Z=(Z_1,Z_2,...Z_n)^T
$$


其似然函数为：  


$$
P(Y|\theta)=\sum_Z{P(Z|\theta)P(Y|Z,\theta)}\\
P(Y|\theta)=\prod_{j=1}^{n}[\pi p^y_{j}(1-p)^{1-y_j}+(1-\pi)q^{y_j}(1-q)^{1-y_j}]
$$



### 目标函数

考虑模型极大似然估计：  


$$
\hat{\theta}=arg\,\max_{\theta}\log{P(Y|\theta)}
$$


EM算法是针对上述无解析解的问题的一种迭代算法  










