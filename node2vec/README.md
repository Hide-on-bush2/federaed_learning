# node2vec读书笔记

## 问题背景

现有的图表征学习的方法学习到的特征都比较固定。举个例子：

<img src="https://tva1.sinaimg.cn/large/00831rSTgy1gdnhq8stbqj30q80ckacw.jpg" width=75% height=75%>

BFS搜索方法倾向于学习一阶临近度，DFS搜索方法倾向于学习二阶临近度。

## 主要贡献

提出了一个介于BFS和DFS搜索方法的搜索方法RandomWalk，该搜索方法可以在搜索一阶临近度和二阶临近度之间做出平衡。

## node2vec神经网络的结构

<img src="https://tva1.sinaimg.cn/large/00831rSTgy1gdni8eb1iwj31db0u0tml.jpg" width=75% height=75%>

## 搜索（采样）方法：RandomWalk

给定一个出发点$u$，然后给定一个搜索的长度$l$，假设$c_i$表示第$i$步搜索到的节点，那么$c_0 = u$。第$i$个被搜索到的节点为$x$的概率为：

$$P(c_i=x|c_{i-1}=v) = \begin{cases}
\frac{\pi_{vx}}{Z} & if(v, x) \in E \\
0 & otherwise
\end{cases}$$

其中$\pi_{vx}$是结点$v$到$x$的转换概率，$Z$是归一化的常量。进一步，$\pi_{vx} = \alpha_{pq}(t, x)·w_{t, x}$，其中

$$\alpha_{pq}(t, x) = \begin{cases}
\frac{1}{p} & if d_{tx} = 0 \\
1 & if d_{tx} = 1 \\
\frac{1}{q} & if d_{tx} = 2
\end{cases}$$

<img src="https://tva1.sinaimg.cn/large/00831rSTgy1gdnitmdci7j30r00f8wgv.jpg" width=75% height=75%>

直观上理解，$p$的值代表着我们希望游走重复遍历已经遍历过的节点的程度。详细来说，如果$p$的值比较小，意味着我们希望游走在每一步选择下一个游走的节点的时候，更倾向于返回上一个节点，将游走局限在一个比较小的范围内。如果$p$的值比较大，意味着更倾向于选择没有遍历过的节点，鼓励扩大搜索范围。  
  
而$q$的值代表着我们是倾向于BFS还是倾向于DFS，是对这两个搜索策略的一个平衡。如果$q$的值比较小，那么搜索策略倾向于DFS；如果$q$的值比较大，那么倾向于BFS。

## 实验结果

<img src="https://tva1.sinaimg.cn/large/00831rSTgy1gdnlmferjqj30pu0pc47y.jpg" width=75% height=75%>





















