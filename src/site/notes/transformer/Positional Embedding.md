---
{"dg-publish":true,"permalink":"/transformer/Positional Embedding/","dgPassFrontmatter":true}
---

# Positional Embedding

Transformer丢失了位置关系，无法区分如A>B和B>A等，所以需要引入位置编码（positional embedding, PE）。

## 位置编码的需求
> 每个token的位置唯一
> 位置编码间距应该固定
> 模型可以适应长短不一的句子
> 位置编码确定
>
> ![pe](/img/user/img/pe.png)
>
> 来自[Transformer 中的 positional embedding - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/359366717#:~:text=关于 pos)

## SinCos位置编码

### **原因**

我们现在需要一种编码表示位置，位置可以看作自然数，自然想到用二进制。
但该编码需要浮点数表示（可能是便于和词向量相加？不太确定），二进制0和1浪费浮点数空间（或者就是不太好相加？而且pe编码要应对不同长度的句子，还需要保持步长一定，不能整数编码完后除以max_len）

所以考虑用正弦函数的组合，近似实现二进制的效果

> “We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $ PE_{pos+k} $ can be represented as a linear function of $ PE_{pos} $.”

> 严格证明
> [Transformer 位置编码中的线性关系 - Timo Denk 的博客 --- Linear Relationships in the Transformer’s Positional Encoding - Timo Denk's Blog](https://blog.timodenk.com/linear-relationships-in-the-transformers-positional-encoding/)

### **位置编码尺寸**

位置编码需要和词向量相加，即shape相同 

> d_model: 词向量的维度
> max_len: 最大的序列长度

```python
self.encoding = torch.zeros(max_len,d_model,device=device)
self.encoding.requires_grad = False
```

### **位置编码公式**

$$
PE_(pos,2i)=sin(pos/10000^{2i/d})
$$
$$
PE_(pos,2i+1)=cos(pos/10000^{2i/d})
$$

其中，pos是PE的位置，从0到max_len
	    _2i是偶数维度的位置，从0到d_model，步长

该函数频率随着向量维度递减（2i增加），每个频率包含一对正弦和余弦

```python
pos = torch.arange(0,max_len,device=device)
pos = pos.float().unsqueeze(dim=1)
_2i = torch.arange(0,d_model,step=2,device=device).float()

self.encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))
self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))
```

```python
# 演示
max_len_try = 4
d_model_try = 10
size = torch.zeros(max_len_try,d_model_try)
print(size)
# max = 4， d_model = 10
# tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

pos_try = torch.arange(0,max_len_try).float().unsqueeze(dim=1)
_2i_try = torch.arange(0,d_model_try,step=2).float()
size[:,0::2] = torch.sin(pos_try/(10000**(_2i_try/d_model_try)))
size[:,1::2] = torch.cos(pos_try/(10000**(_2i_try/d_model_try)))
print(size)
# tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,
#           1.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00,  1.0000e+00],
#         [ 8.4147e-01,  5.4030e-01,  1.5783e-01,  9.8747e-01,  2.5116e-02,
#           9.9968e-01,  3.9811e-03,  9.9999e-01,  6.3096e-04,  1.0000e+00],
#         [ 9.0930e-01, -4.1615e-01,  3.1170e-01,  9.5018e-01,  5.0217e-02,
#           9.9874e-01,  7.9621e-03,  9.9997e-01,  1.2619e-03,  1.0000e+00],
#         [ 1.4112e-01, -9.8999e-01,  4.5775e-01,  8.8908e-01,  7.5285e-02,
#           9.9716e-01,  1.1943e-02,  9.9993e-01,  1.8929e-03,  1.0000e+00]])
```

![image-20241007150641294](/img/user/img/image-20241007150641294.png)

看下图纵轴Sequence Position(max_len)，每个值（即句子中token的位置（pos））各不相同

![image-20241007145605447](/img/user/img/image-20241007145605447.png)

下图同理（sin和cos分开绘制）
![29be0ec4e496473feacdefeac4df0c8b](/img/user/img/29be0ec4e496473feacdefeac4df0c8b.png)

​										图片来源 The Illustrated Transformer

### **完整代码**

```python
class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len,device):
        # d_model: 词向量的维度
        # max_len: 最大的序列长度
        super(PositionalEmbedding,self).__init__()
        self.encoding = torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad = False
        # pos是位置编码的位置，从0到max_len
        pos = torch.arange(0,max_len,device=device)
        pos = pos.float().unsqueeze(dim=1)
        # _2i是偶数维度的位置，从0到d_model，步长
        _2i = torch.arange(0,d_model,step=2,device=device).float()
        self.encoding[:,0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2] = torch.cos(pos/(10000**(_2i/d_model)))
        
        # 绘图
        plt.figure(figsize=(10, 8))
        plt.imshow(self.encoding, aspect='auto', cmap='coolwarm')
        plt.colorbar(label="Positional Encoding Values")
        plt.xlabel("Embedding Dimension(d_model)")
        plt.ylabel("Sequence Position(max_len)")
        plt.title(f"Positional Encoding (d_model={d_model}, max_len={max_len})")
        plt.show()
        
	def forward(self,x):
        batch_size,seq_len = x.size()
        return self.encoding[:seq_len,:] 
        
PositionalEmbedding(40,20,device='cpu')
```

### **缺失内容**

positional Embedding是如何训练的

为什么PE和词向量是相加的



参考：Transformer 中的 positional embedding - Felixue的文章 - 知乎
https://zhuanlan.zhihu.com/p/359366717

[The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time. (jalammar.github.io)](https://jalammar.github.io/illustrated-transformer/)