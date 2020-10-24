# Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies

## 背景

现有的关于联邦学习的收敛性证明已经囊括了Full Participation和Partical Participation的收敛性证明，但是这些工作都是默认选择到的设备都是无偏差的情况(unbiased)，这里的偏差指的是相同的全局模型在不同的设备上的不同表现。现有的联邦学习收敛性证明工作都是用一个差值上界来统一这些差异。这篇文章指出，如果在每一轮选择训练的设备的时候偏向于选择具有较高损失的设备，可以得到更快的收敛速度

## 论文贡献

* 提出了考虑到设备偏差的第一个收敛性证明
* 提出POWER-OF-CHOICE设备选择机制