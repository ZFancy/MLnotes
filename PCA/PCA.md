---
title: PCA
tags: Machine_learning
show_subscribe: false
author: Zhu Jianing
---

Principal component analysis (PCA) 主成分分析：一种常用的无监督学习方法

<!--more-->

| 类别 | 算法 |   主要模型   | 求解方向 | 解法 |
| :--: | :--: | :----------: | :------: | :--: |
| 降维 | PCA  | 低维正交空间 | 方差最大 | SVD  |

### 定义简介

$$
x = (x_1,x_2,..,x_m)^T
$$


$$
\mu = E(x) = (\mu_1,\mu_2,...\mu_m)^T
$$

$$
\sum = cov(x,x)=E[(x-\mu)(x-\mu)^T]
$$

$$
y_k=a_{k}^Tx=a_{1k}x_1+a_{2k}x_1+...+a_{mk}x_m,\,\ k=1,2,...,m
$$

$$
var(y_k)=a_{k}^T\sum a_k=\lambda_k,\,\ k=1,2,...,m
$$

$$
\max_{a_1}\,\ a_{1}^T\sum a_{1}\\
s.t.\,\ a_{1}^Ta_{1}=1
$$

$$
a_{1}^T\sum a_{1}-\lambda (a_{1}^Ta_{1}-1)
$$

$$
\sum a_1-\lambda a_1 = 0
$$
因此 $\lambda$ 是 $\sum$ 的特征值， $a_1$ 是对应的单位特征向量

之后部分证明省略

得出定理

> 第k主成分的方差等于协方差矩阵的第k个特征值，而第k主成分既原始向量于对应的特征向量上的线性组合



### 主成分个数

第k主成分的方差贡献率定义为其方差与所有方差之和的比，累计方差贡献率是k个主成分累计方差之和与所有方差之和的比 


$$
\eta_k = \frac{\lambda_k}{\sum_{i=1}^m\lambda_i}
\\
\sum_{i=1}^k\eta_i=\frac{\sum_{i=1}^k\lambda_i}{\sum_{i=1}^m\lambda_i}
$$


通常取k使累计方差贡献率达到规定的百分比如70%~80%以上



### 回顾整理两种PCA的主要算法

#### 相关矩阵的特征值分解算法

1. 规范化数据矩阵 $X$ 



$$
x_{ij}^*=\frac{x_{ij}-\overline x_i}{\sqrt{s_{ii}}}\,\ ,\,\ i=1,2,...,m;\,\ j=1,2,...,n
$$

$$
\overline{x_i}=\frac{1}{n}\sum_{j=1}^nx_{ij},\,\ i=1,2,...,m\\
s_{ii}=\frac{1}{n-1}\sum_{j=1}^n(x_{ij}-\overline{x_i})^2,\,\ i=1,2,...,m
$$



2. 依据规范化数据矩阵计算 $R$  


$$
R=[r_{ij}]_{m\times m}=\frac{1}{n-1}XX^T
$$

$$
r_{ij}=\frac{1}{n-1}\sum_{l-1}^nx_{il}x_{lj},\,\ i,j=1,2,...,m
$$



3. 求取样本相关矩阵 $R$ 的 $k$ 个特征值和对应的单位特征向量，其中按照方差贡献率来求需要的主成分个数，确定最终的单位特征向量


$$
|R-\lambda I|=0\\
\lambda_1\geq\lambda_2\geq,...,\geq\lambda_m\\
$$

$$
\lambda_1,\lambda_2,...,\lambda_k\\
a_i=(a_{1i},a_{2i},...,a_{mi})^T,\,\ i=1,2,...,k
$$



4. 求样本主成分


$$
y_i=a_{i}^Tx,\,\ i=1,2,...,k
$$


#### 数据矩阵的奇异值分解算法

利用低秩奇异值分解（截断奇异值分解）进行主成分分析。

1. 给定 $m\times n$ 的矩阵 $X$ ，定义新型矩阵如下


$$
X'=\frac{1}{\sqrt{n-1}}X^T
$$


2. 对新型矩阵进行奇异值分解 


$$
X'=U\sum V^T
$$


3. 求 $k\times n$ 样本主成分矩阵 


$$
Y=V^TX
$$


