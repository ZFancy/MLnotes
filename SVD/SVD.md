---
title: SVD
tags: Machine_learning
show_subscribe: false
author: Zhu Jianing
---

Singular value decomposition (SVD) 奇异值分解：矩阵因子分解法，部分机器学习算法中核心部分。

<!--more-->

### 定义简介

> 矩阵的奇异值分解指，将一个非零的 $m\times n$ 实矩阵 $A，A\in R^{m\times n}$ 表示为以下三个实矩阵乘积形式的运算，进行矩阵的因子分解


$$
A = U\sum V^T\\
$$

$$
UU^T,\,\ VV^T\\
\sum =diag(\sigma_1,\sigma_2,...,\sigma_k)\\
k=min(m,n)\\
\sigma_1\geq \sigma_2\geq ... \sigma_p \geq0
$$



> 其中 $U $ 是 $m$ 阶正交矩阵，$V$ 是 $n$ 阶正交矩阵，而 $\sum$ 是由**降序排列的非负的对角线元素**组成的 $m\times n$ 矩形对角矩阵，满足上述条件。
>
> **$U\sum V^T$** 为矩阵的**奇异值分解**，对角矩阵对角线上的值为矩阵的**奇异值**，$U$ 的列向量为左奇异向量，$V$ 的列向量为右奇异向量 



- 奇异值分解是一种用因子分解的方式近似表示原始矩阵的方法，也可以看作是矩阵形式数据压缩的一种方法。

- 由奇异值分解基本定理可知：实矩阵的奇异值分解一定存在，但不唯一。

证明部分由构造性方法完成，由初始矩阵 $A$ 构造 $U ,\sum, V$ 继而证明 $A = U\sum V^T$ ，此处省略。

以上给出的是完全奇异值分解定义，而实际常用的是奇异值分解紧凑形式和截断形式，前者 $\sum$ 与原始矩阵等秩，后者 $\sum$ 比原始秩低，从数据压缩的角度来看，前者对应于无损压缩，而后者对应着的是有损压缩。



### 等秩奇异值分解

也称作紧奇异值分解：

> 设 $m\times n$ 实矩阵 $A$ , 矩阵秩为 $rank(A)=r, r\leq min(m,n)$ ，则称 $U_r\sum_r V_{r}^T$ 为紧奇异值分解 （compact singular value decomposition）
>
> $U_r$ : $U_{m\times r}$ ,由完全奇异值分解的 $U$ 前 $r$ 列组成 
>
> $V_r$ :  $V_{r\times n}$ ,由完全奇异值分解的 $V$ 前 $r$ 列组成 
>
> $\sum_r$ :  $\sum_{r\times r}$ ,由完全奇异值分解的 $\sum$ 前 $r$ 个对角线元素组成 



### 低秩奇异值分解

也称截断奇异值分解，区别于等秩的分解情况，该形式的分解只是近似 即 $A\simeq U_k\sum_k V_{k}^T$ 而非完全一致：

> 设 $m\times n$ 实矩阵 $A$ , 矩阵秩为 $rank(A)=r, 0<k<r$ ，则称 $U_r\sum_r V_{r}^T$ 为截断奇异值分解 （truncated singular value decomposition）
>
> $U_r$ : $U_{m\times r}$ ,由完全奇异值分解的 $U$ 前 $r$ 列组成 
>
> $V_r$ :  $V_{r\times n}$ ,由完全奇异值分解的 $V$ 前 $r$ 列组成 
>
> $\sum_r$ :  $\sum_{r\times r}$ ,由完全奇异值分解的 $\sum$ 前 $r$ 个对角线元素组成 



### 几何意义

令自己比较感叹的一块，是对于奇异值分解的几何意义解释上：

简单来看，$m\times n$ 的实矩阵 $A$ 表示了由 $n$ 维空间 $R^n$ 到 $m$ 维空间 $R^m$ 的线性变换，  


$$
x\rightarrow Ax
$$


其中这个线性变换也可以分解为对应奇异值分解的三个矩阵的不同变换

- $V^T$ 坐标系的旋转或反射变换
- $U$ 坐标系的旋转或反射变换
- $\sum $ 坐标轴的缩放变换

因为 $U , V^T$ 均为正交矩阵，而 $\sum $ 为对角形式矩阵。

继而可以讨论下奇异值分解的一些相关性质（此处为完全奇异值分解）：

1. 矩阵 $A^TA,AA^T$ 的奇异值分解可由 $A$ 的分解来表示，如其中： 


$$
A^TA=(U\sum V^T)^T(U\sum V^T)=V({\sum}^T\sum)V^T
$$


2. 前面提及分解不唯一指的是两个正交矩阵不唯一，而一个实矩阵的奇异值分解中，其奇异值是唯一的；
3. 完全奇异值分解的矩阵秩与 $\sum$ 的秩相等，即奇异值的个数 ；
4. 将性质1中的基础公式变形得：


$$
AV=U\sum
$$

$$
Av_j=\sigma_ju_j,\,\ j=1,2,...,n\\
A^Tu_j=\sigma_jv_j,\,\ j=1,2,...,n\,\,\,\,\ A^Tu_j=0,\,\ j=n+1,n+2,...,m
$$



### 奇异值分解的计算

由以上性质不难发现，矩阵 $A$ 的奇异值分解可以通过求对称矩阵 $A^TA$ 的特征值与特征向量得到，其特征向量构成正交矩阵 $V$ 的列，通过求 $AA^T$ 得特征向量，构成正交矩阵 $U$ 的列，而特征值与奇异值关系如下： 


$$
\sigma_j=\sqrt{\lambda_j},\,\ j=1,2,...,n
$$


步骤归纳：

1. 求 $A^TA$ 的特征值与特征向量，特征向量单位化后构成 $V$ ；
2. 求 $A A^T$ 的特征值与特征向量，特征向量单位化后构成 $U$ ；
3. 用上述所得特征值求平方，构成 $\sum$ ;

其中 Python 中的 Numpy 已经有构造好的函数可以进行矩阵的奇异值分解，

~~~python
import numpy as np

#Show a matrix
A = np.array([[1,1],[2,2],[0,0]])

#Use numpy.linalg.svd() to get U,D,V
U,D,V = np.linalg.svd(A)

~~~

对图像进行奇异值分解可以更加直观地感受，

初始图像大小为（512 $\times$ 512）

<img width="190" height="190" src="https://raw.githubusercontent.com/ZFancy/ZFancy.github.io/master/assets/images/1.jpg"/>

使用不同个数的奇异值分解结果如下：

<img width="600" height="400" src="https://github.com/ZFancy/ZFancy.github.io/blob/master/assets/images/result.png?raw=true"/>

~~~python
import numpy as np 
from PIL import Image
import math 
import os

#Function of SVD 
def SVD(sigma, u, v, k):
    m = len(u)
    n = len(v[0])
    a = np.zeros((m, n))
    for kk in range(k):
        uu = u[:,kk].reshape(m, 1)
        vv = v[kk].reshape(1, n)
        a = a + sigma[kk] * np.dot(uu, vv)
    a = a.clip(0, 255)
    return np.rint(a).astype('uint8')

#Read image
A = Image.open("1.jpg",'r')
a = np.array(A)
print('Image.size: ',a.shape)

#Deal with R G B matrixes
ur,sigmar,vr = np.linalg.svd(a[:, :, 0])
ug,sigmag,vg = np.linalg.svd(a[:, :, 1])
ub,sigmab,vb = np.linalg.svd(a[:, :, 2])

Num = 20
for k in range(1,Num+1):
    R = SVD(sigmar, ur, vr, k)
    G = SVD(sigmag, ug, vg, k)
    B = SVD(sigmab, ub, vb, k)
    new_image = np.stack((R, G, B), axis=2)
    Image.fromarray(new_image).save('SVD_out\svd_%d.jpg' % k)

~~~

有时候简单实现下学习的算法，收获还是很多的  :)



### 矩阵近似

奇异值分解是在平方损失意义下对矩阵的最优近似。

引入 Frobenius norm ，矩阵的 $L_2$ 范数直接推广，对应着机器学习中的平方损失，

> 设矩阵 $A \in R^{m\times n}$ , $A=[a_{ij}]_{m\times n}$ ，其  Frobenius norm 为 
> $$
> ||A||_F=(\sum_{i=1}^m\sum_{j=1}^n(a_{ij})^2)^{\frac{1}{2}}
> $$

其中矩阵 $A$ 的奇异值分解可得 $\sum=diag(\sigma_1,\sigma_2,...,\sigma_n)$ 则可证明有：


$$
||A||_F=(\sigma_1^2+\sigma_2^2+...+\sigma_n^2)^{\frac{1}{2}}
$$
在秩不超过原始矩阵的矩阵集合中，存在矩阵 $A$ 的 Frobenius norm 意义下的最有近似矩阵 $X$ ，使得： 


$$
||A-X||_F=(\sigma_1^2+\sigma_2^2+...+\sigma_n^2)^{\frac{1}{2}}
$$


利用外积展开式可求对矩阵 $A$ 的近似，矩阵 $A$ 的奇异值分解也可以由外积形式表示为 $U\sum$ 和 $V^T$ 的乘积，即将前者按列向量分块， 


$$
U\sum=[\sigma_1u_1\,\ \sigma_2u_2\,\,...\,\ \sigma_nu_n]
$$


后者按行向量分块，  


$$
V^T=\left[\begin{matrix} v_1^T\\v_2^T\\...\\v_n^T \end{matrix}\right]
$$


则可得矩阵最优近似，


$$
A=\sigma_1u_1v_1^T+\sigma_2u_2v_2^T+...+\sigma_nu_nv_n^T
$$


其中 $n$ 可设定为具体秩的值。