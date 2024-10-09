---
{"dg-publish":true,"permalink":"/transformer/attention/","dgPassFrontmatter":true}
---

# Attention
[[Transformer.canvas|Transformer]]

> 按照查询，索引，值，一直没完全理解QKV
>
> 后来看到王木头学科学【从编解码和词嵌入开始，一步一步理解Transformer，注意力机制(Attention)的本质是卷积神经网络(CNN)】 https://www.bilibili.com/video/BV1XH4y1T76e/?share_source=copy_web&vd_source=1d5fa273132a8f9ff2499d407fcb2175，换种思路，好像抓住了一点。
>
> 本文图片大多来自上述视频

> 下文有点混淆token和词的关系，不是很严谨（写完才发现，改不动了

> Embedding可以看作字典，是单个token的语义。
>
> 通过注意力机制，理解token组合后的语义。

## **数学公式**

$$
Attention(Q,K,V)=Softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V
$$

$$
\begin{aligned} Q(Query) \quad = \quad W^{Q}X \\
 K(Key) \quad =  \quad W^{K}X \\
 V(Value) \quad = \quad W^{V}X \end{aligned}
$$

Attention示意图

![image-20241008231521832](https://notefangpenglai.oss-cn-beijing.aliyuncs.com/photo/202410082315912.png)

来自 【从编解码和词嵌入开始，一步一步理解Transformer，注意力机制(Attention)的本质是卷积神经网络(CNN)】

## QKV的理解

### $X$**经过线性层得到$QKV$**

> 线性层（Linear Layer），全连接层
> 初始随机化，不断训练得到适合的权重

不使用原本的X，而经过$W_{q}$，$W_{k}$，$W_{v}$，X经过线性变化，增强模型的拟合能力。

### $Q$ 和$K^{T}$**的点积**

> **点积**        
> 计算向量与向量间的相关联程度

计算一个token，与上下文其余token的关系

> 假设“……梦里若有，梦是反的：梦里若无，唯梦闲人不梦君……”（小小伤感一下（不是）

那么“君”代之谁，需要在前后文中查找，
而且前后文可能出现多个人物，就需要找到一个和“君”在语义空间中最相邻的人称向量。

> <img src="https://notefangpenglai.oss-cn-beijing.aliyuncs.com/photo/202410090006139.png" alt="image-20241009000622813" style="zoom: 50%;" />
> 
>通过上述过程，得到了矩阵A$(T*T)$，进行scale，然后对每一行进行Softmax（应该是得到了文章中每一个token和全部token的相关性，或者重要程度？）

### **$\sqrt{d_{K}}$**（scale）**的理解**

<img src="https://notefangpenglai.oss-cn-beijing.aliyuncs.com/photo/202410082328154.png" alt="image-20241008232812083" style="zoom: 50%;" />

除以$\sqrt{d_{K}}$，防止输入softmax的值过大（数值集中在softmax的两侧，梯度较小的地方），梯度消失。

> 因为softmax，即概率方向考虑：
> 假设$X$和$Y$服从多元正态分布，每一项相互独立且服从标准正态分布$(0,1)$
> 计算$XY^{T}$后其服从$(0,D_{out})$，通过除以标准差$\sqrt{d_{K}}$，使得方差变为1

## **$QK^{T}$乘以$V$**

> 前情提要（
>
> 假设“……梦里若有，梦是反的：梦里若无，唯梦闲人不梦君……”（小小伤感一下（不是）
> 那么“君”代之谁，需要在前后文中查找，而且前后文可能出现多个人物，就需要找到一个和“君”在语义空间中最相邻的人称向量。

在$V$中（假设$V$是单独的token词典，相当于新华字典），“君”在语义空间内可能和“人”相接近。

那么可能上述文章模仿、接续了屈原等“香草美人”的文化内涵 （我承认是瞎编的（），“君”不只是“需要凭流水通辞的佳人”，也可是一点点微不足道的“理想”。

所以需要对“新华字典”做修正，使“君”更符合全文的含义。

<img src="https://notefangpenglai.oss-cn-beijing.aliyuncs.com/photo/202410090041528.png" alt="image-20241009004144416" style="zoom:50%;" />

假设$t_2$对应的token是“君”，

在一个$(len*len)$(len指token的数量)的矩阵中，假设$\widehat{t_i}$指的是“人”,数值为0.3，$\widehat{t_j}$值得是“理想”，数值为0.4。

现在对$V$修正

> 词典 d_model 语义的维度，现假设该维度有明确的含义，经过softmax后，数据集中在-1，1中
> 再假设数据越接近1，词义越靠近该维度

将$t_2$行乘$V$中第二列（假设第二列是“理想”的维度）（乘一整列，浏览了整个数据）

> 假设原位置数据为0.2，现在修正后数据为0.6，可以说注意到了"君"在本文的含义。





参考：【从编解码和词嵌入开始，一步一步理解Transformer，注意力机制(Attention)的本质是卷积神经网络(CNN)】 https://www.bilibili.com/video/BV1XH4y1T76e/?share_source=copy_web&vd_source=1d5fa273132a8f9ff2499d407fcb2175

transformer中的attention为什么scaled? - 欠拟合的回答 - 知乎
https://www.zhihu.com/question/339723385/answer/3262049069

[注意力机制到底在做什么，Q/K/V怎么来的？一文读懂Attention注意力机制 (zhihu.com)](https://www.zhihu.com/tardis/zm/art/414084879?source_id=1005#:~:text=输入矩阵 X)

