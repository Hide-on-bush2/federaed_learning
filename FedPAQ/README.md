# FedPAQ: A Communication-Efficient Federated Learning Method with Periodic Averaging and Quantization

## 论文的贡献

这篇论文的贡献主要是在传统的联邦学习方法上做了3个改进：
* 定期更新(Periodic averaging)
* 部分节点参与(Partical node participation)
* 低精度量化(Low-precision quantizer)

其中第一和第二点在很多其他的论文中也出现过，第三点也许是一个不错的切入点

## FedPAQ算法

![](https://i.loli.net/2020/10/18/PRyIghoHL7YNkuB.png)

这个算法与我们熟悉的联邦学习方法的不同点在于：在每次进行全局更新的时候不是直接将$X_{k, r}^{(i)} - X_k$上传，而是先对它们进行一个量化$Q(X_{k, r}^{(i)} - X_k)$，再将这个参数进行上传

这种量化的方法可以有很多种，这篇论文用的是一种低精度量化

### 低精度量化(Low-precision quantizer)

![](https://i.loli.net/2020/10/18/PrhKDB2IXMO1C4Q.png)

在这里，$x_i$表示的是$X$矩阵中的元素，简单地理解，这个映射关系是将矩阵元素的绝对值要么变成$\frac{l+1}{s}$，要么变成$\frac{l}{s}$，这个会根据元素的绝对值与矩阵的二范数$||x||$所组成的一个概率$\frac{|x_i|}{||X||}s - l$来决定

## 收敛性

这篇文章给出了这种训练方式的一个收敛上界：
![](https://i.loli.net/2020/10/19/Su1QtOdgwXmCPMF.png)

证明过程就不在这里放出来了

## 实验

### 设置1

![](https://i.loli.net/2020/10/19/V54vjztFSk8ugsB.png)

可以看出三点：
* 使用量化方法要比不使用量化方法更快地收敛
* 量化方式中的一个超参数$s$的值越大，收敛速度会越快
* 最后训练出来的模型性能并没有什么参数

### 设置2

![](https://i.loli.net/2020/10/19/qM69yLGtJAZC5be.png)

可以看到这篇论文介绍的方法FedPAQ比FedAvg和SDG方法的收敛速度更快

## 结论

这篇文章整体上的亮点不多，但我觉得提供了一种新的切入角度：考虑设备发送给服务器的参数，可以设计一些合适的量化方式，来达到我们的一些目的：减少上传参数的大小、加快收敛速度、得到更好的收敛上界，甚至可以使得联邦训练的方式更加公平

目前为止，我了解到的对传统联邦学习进行改进的角度有：
* 改变Average的方式
* 改变全局的损失函数
* 对设备上传的参数进行量化
* 服务器进行Sample的方式

