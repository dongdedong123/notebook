---
{"dg-publish":true,"permalink":"/transformer/attention/","dgPassFrontmatter":true}
---

# Attention

## **数学公式**

$$
Attention(Q,K,V)=Softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V
$$
$$
\begin{aligned} Q(Query) \quad = \quad W^{Q}X \\
 K(Key) \quad =  \quad W^{K}X \\
 V(Value) \quad = \quad W^{V}X \end{aligned}
$$

## **理解**

### $X$**经过线性层得到$QKV$**

> 线性层（Linear Layer），全连接层
> 初始随机化，不断训练得到适合的权重
>
> <img src="https://notefangpenglai.oss-cn-beijing.aliyuncs.com/photo/202410231750675.png" alt="image.png" style="zoom: 50%;" />
>
> > 来自：[LLM Visualization](https://bbycroft.net/llm) 没有看到许可证（

不使用原本的X，而经过$W_{q}$，$W_{k}$，$W_{v}$，X经过线性变化，增强模型的拟合能力。

###   $Q$ 和$K^{T}$**的点积**

> **点积**        
> 计算向量与向量间的相关联程度

计算一个token，与上下文其余token的关系

> <img src="https://notefangpenglai.oss-cn-beijing.aliyuncs.com/photo/202410090006139.png" alt="image-20241009000622813" style="zoom: 50%;" />
>
> > 来自： 【从编解码和词嵌入开始，一步一步理解Transformer，注意力机制(Attention)的本质是卷积神经网络(CNN)】 https://www.bilibili.com/video/BV1XH4y1T76e/?share_source=copy_web&vd_source=1d5fa273132a8f9ff2499d407fcb217
>
> > 通过上述过程，得到了矩阵A$(T*T)$，进行scale，然后对每一行进行Softmax（应该是得到了文章中每一个token和全部token的相关性，或者重要程度？）

### **$\sqrt{d_{K}}$**（scale）**的理解**

<img src="https://notefangpenglai.oss-cn-beijing.aliyuncs.com/photo/202410231804802.png" alt="image-20241008232812083" style="zoom: 50%;" />

除以$\sqrt{d_{K}}$，防止输入softmax的值过大（数值集中在softmax的两侧，梯度较小的地方），梯度消失。

> 因为softmax，即概率方向考虑：
> 假设$X$和$Y$服从多元正态分布，每一项相互独立且服从标准正态分布$(0,1)$
> 计算$XY^{T}$后其服从$(0,D_{out})$，通过除以标准差$\sqrt{d_{K}}$，使得方差变为1

### $QKV$**的理解**

#### **王木头的解释：**

> 假设“……唯梦闲人不梦君……”（小小伤感一下（不是）
> 那么“君”代指何人或何物，需在前后文中查找。
> 但前后文可能出现多个有几率指代的人或物
> 需找到一个和“君”在语义空间中最相邻的人或物的向量。

在$V$中（假设$V$是单独的token词典，相当于新华字典），“君”在V的向量空间中内可能和“人”相接近。

通过上下文发现“君”和A相近

对“新华字典”（V）修正，使“君”符合全文含义。

<img src="https://notefangpenglai.oss-cn-beijing.aliyuncs.com/photo/202410231805987.png" alt="image-20241009004144416" style="zoom:50%;" />

**总结：**
	V是字典，需要一个矩阵对V修正。
	选择两个不同的矩阵Q和K，转置相乘后为二次型，非线性，增强模型的拟合能力
	至于Q和K的实际含义，一个可以看作主观语义，另一个看作客观语义

#### **查询—键值对的理解**

> google嘛 擅长搜索引擎（

Q(query) 搜索信息是输入的内容
K(key)搜索出来的网页标题
V(value)我们希望获取的内容

（K和V是一一对应的，类似于键值对的关系）

> $Q$ 和 $K$ 相乘，得到 $Q$ 和 $K$ 的相似度得分（**可能 $Q$ 的表述是多样的，对应到已有的 $K$ 上**）
> 那么相似度得分再和 $value$ 乘积（经过softmax，$QK^{T}$就可以看作全部的概率（并行计算，矩阵相乘））
> $$
> P(输出)=	\sum{V_{i}}*P(i与Q的关系)
> $$
>

#### 联系

其实理解是类似的。**王木头**中客观词义相当于键值对的K，主观词义相当于输入的Q，把**修正**当作**查询**，

就是**查询—键值对的理解**





参考：【从编解码和词嵌入开始，一步一步理解Transformer，注意力机制(Attention)的本质是卷积神经网络(CNN)】 https://www.bilibili.com/video/BV1XH4y1T76e/?share_source=copy_web&vd_source=1d5fa273132a8f9ff2499d407fcb2175

transformer中的attention为什么scaled? - 欠拟合的回答 - 知乎
https://www.zhihu.com/question/339723385/answer/3262049069

[注意力机制到底在做什么，Q/K/V怎么来的？一文读懂Attention注意力机制 (zhihu.com)](https://www.zhihu.com/tardis/zm/art/414084879?source_id=1005#:~:text=输入矩阵 X)

[Transformer 架构中的 query、key 和 value 是什么，为什么使用它们？|易卜拉欣·皮什卡 --- What are Query, Key, and Value in the Transformer Architecture and Why Are They Used? | Ebrahim Pichka (epichka.com)](https://epichka.com/blog/2023/qkv-transformer/)[小白都能看懂的超详细Attention机制详解 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/380892265)

[变压器解释器：LLM 变压器模型直观解释 --- Transformer Explainer: LLM Transformer Model Visually Explained](https://poloclub.github.io/transformer-explainer/)

[LLM Visualization](https://bbycroft.net/llm)