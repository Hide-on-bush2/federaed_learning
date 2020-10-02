# ON THE CONVERGENCE OF FEDAVG ON NON-IID DATA

这篇论文给出了不需要分布数据集是IID的假设的一个收敛上界

## 背景

现有的联邦学习关于收敛性证明的工作需要有以下两个假设：

* 分布在不同的设备上的数据集是IID的
* 参与联邦学习的每一个设备都能与服务器保持稳定的连接

事实上这两个假设在现实部署的过程中是难以实现的，首先，保证不同设备上的数据集都是IID的显然是难以保证的，其次，我们也无法保证参与联邦学习的每一个设备都时刻保持有效的连接，当某一台设备关闭电源或者断开连接的是否，那么服务器和其他设备都需要等待这一台的重启或重新连接，这明显会造成很大的耗费

对于第二个问题的解决办法是每次服务器向设备广播数据集的时候都会选择其中保持连接的一部分设备进行广播然后进行训练，而不是向所有的设备进行广播

## 贡献

该论文主要有以下两个贡献：

* 给出了不需要以上两个假设的两个收敛上界，分别包括Full Device Participation(每次都选择所有的设备来训练)和Partial Device Participation(每次只选择一部分的设备进行训练)
* 提出全局模型更新的步长需要衰减，并给出解释

## 收敛上界

这篇论文给出的收敛上界需要满足以下4个假设：

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjb45a7t9qj31dy0780uk.jpg)

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjb45ptrwtj31dk08i419.jpg)

然后它给出的Full Device Participation收敛上界是：

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjb46e8xchj31eq0egjue.jpg)

给出的Partial Device Participation收敛上界是：

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjb4acwu25j31ds0doaei.jpg)

## 步长衰减

论文支持，学习率的下降对于非IID环境下FedAvg的效果直观重要，他们提出了下面一条定理：

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1gjb4espxesj31ga0bujum.jpg)

除此之外，还有一些关于步长对于FedAvg收敛效果的讨论：

* 定理1告诉我们，当$E > 1$并且学习步长递减的时候，FedAvg会逐渐收敛到最优
* 定理4告诉我们，当$E > 1$并且学习步长固定的时候，FedAvg不会收敛到最优

## 结论

这篇论文给出了不需要数据集为IID的假设的收敛上界，并且关于步长对于收敛性的影响进行了讨论，我觉得是一篇比较有开创性的文章。相关的数学证明我还会继续去研究以下

