> VAE中的encoder改成固定的添加高斯噪声的过程

原理总结可以参考这篇博客：[https://shao.fun/blog/w/how-diffusion-models-work.html](https://shao.fun/blog/w/how-diffusion-models-work.html)
<a name="J5roI"></a>
## 原始论文
> [Deep unsupervised learning using nonequilibrium thermodynamics](http://proceedings.mlr.press/v37/sohl-dickstein15.html)

$x^0$为原始图片，$q$表示前向的分布（先验），$p$表示后向的分布（后验），$\pi(y)$为期望的最终分布（tractable的）。
<a name="JYKns"></a>
### 前向过程
前向过程的条件概率：$q(x^t\mid x^{t-1})\triangleq T_\pi(x^t\mid x^{t-1};\beta_t)$其中$\beta_t$为diffusion rate（模型参数）<br />前向过程的轨迹：$q(x^{0:T})=q(x^0)\prod_{i=1}^{T}q(x^t\mid x^{t-1})$
<a name="uQqqe"></a>
### 反向过程
反向过程的轨迹：$p(x^{0:T})=p(x^T)\prod_{i=1}^{T}p(x^{t-1}\mid x^{t})$，其中$p(x^T)=\pi(x^T)$<br />在极限情况下（$\beta_t\to 0^+,T\to+\infty$），当q为高斯/二项分布，p也可以证明为高斯/二项分布
<a name="ITLk5"></a>
### loss的推导
> 只考虑高斯分布，文章讨论的二项分布暂不归纳，并且K我对文章中定义的取了负

1. 假定$p(x^{t-1}\mid x^{t})=\mathcal{N}(\mu_\theta(x^t,t),\Sigma_\theta(x^t,t))$，模型学习的就是均值μ和方差Σ，本文考虑用MLP拟合。
2. 模型给数据生成的概率分布是$p(x^0)=\int p(x^{0:T})d{x^{1:T}}$（所有step的特征联合概率分布，对1:T step的特征积分）
3. 做如下转换，与前向概率相联系，使公式变的tractable：

$\begin{aligned}
p(x^0)
&=
\int p(x^{0:T})\frac{q(x^{1:T}\mid x^{0})}{q(x^{1:T}\mid x^{0})} dx^{1:T}\\
&=
\int \frac{p(x^{0:T})}{q(x^{1:T}\mid x^{0})} \;q(x^{1:T}\mid x^{0}) dx^{1:T}\\
&=
\int p(x^T)\prod_{t=1}^T\frac{p(x^{t-1}\mid x^t)}{q(x^{t}\mid x^{t-1})} \;q(x^{1:T}\mid x^{0})dx^{1:T}\\
\end{aligned}$

4. 给定一个样本$x^0$通过多次前向，对所有轨迹求均值，就能得到积分$q(x^{1:T}\mid x^{0})dx^{1:T}$的近似值（即蒙特卡洛方法），p用MLP计算，q用预定义的高斯分布计算。
5. 实操中只需要运行一次前向，因为在极限情况下，正反向轨迹可以被弄成相同，从而一次前向就是对积分完全准确的估计（这个性质我没完全懂）
6. 训练的目标是，最小化p与q的交叉熵：$L=\int -\log p(x^0)\;q(x^0)dx^0$。也就是把所有样本用p求出来的概率负对数求均值
7. 使用Jensen不等式求L的下界（积分的log小于等于log的积分），我理解这一步避免了需要求积分的log（和的log是不可以拆分的）：

$\begin{aligned}
L&=\int -\log \left(\int p(x^T)\prod_{t=1}^T\frac{p(x^{t-1}\mid x^t)}{q(x^{t}\mid x^{t-1})} \;q(x^{1:T}\mid x^{0})dx^{1:T}\right)\;q(x^0) dx^0 \\
&\geq
\int -\log \left( p(x^T)\prod_{t=1}^T\frac{p(x^{t-1}\mid x^t)}{q(x^{t}\mid x^{t-1})} \right)q(x^{0:T}) dx^{0:T}\\
&=
\int-\log \left( \prod_{t=1}^T\frac{p(x^{t-1}\mid x^t)}{q(x^{t}\mid x^{t-1})} \right)q(x^{0:T}) dx^{0:T}
+
\int -\log p(x^T) q(x^T) dx^T
\end{aligned}$

8. 优化目标为$K=\sum_{t=1}^T\int-\log \frac{p(x^{t-1}\mid x^t)}{q(x^{t}\mid x^{t-1})} q(x^{0:T}) dx^{0:T}+H_p(X^T)$，最后一项是常数（$x^T$的分布为固定的正态分布）
9. 为了减小边界效应（原文：to avoid edge effects），文章假设$p(x^0\mid x^1)=q(x^1\mid x^0)$，从而$K=\sum_{t=2}^T\int-\log \frac{p(x^{t-1}\mid x^t)}{q(x^{t}\mid x^{t-1})} q(x^{0:T}) dx^{0:T}+H_p(X^T)$。为什么要这么做我没有理解，猜测是$x^0$是离散的确定值，不希望生成出训练集的图片？
10. 重写q为后验分布（因为markov分布与历史状态无关，这里条件概率可以先多写一个$x^0$，让$q(x^{t-1}\mid x^t)$变得可以计算，这可能也是上一步要把第一项去掉的原因，不然0的位置后验不容易算）：

$\begin{aligned}
K&=
\sum_{t=2}^T\int-\log \frac{p(x^{t-1}\mid x^t)}{q(x^{t}\mid x^{t-1},x^0)} q(x^{0:T}) dx^{0:T}+H_p(X^T)\\
&=
\sum_{t=2}^T\int-\log \frac{p(x^{t-1}\mid x^t)}{q(x^{t-1}\mid x^{t},x^0)}\frac{q(x^{t-1}\mid x^0)}{q(x^t\mid x^0)} q(x^{0:T}) dx^{0:T}+H_p(X^T)\\
&=
\sum_{t=2}^T\int-\log \frac{p(x^{t-1}\mid x^t)}{q(x^{t-1}\mid x^{t},x^0)}q(x^{0:T}) dx^{0:T} + \sum_{t=2}^{T}\left(H_q(X^{t-1}\mid X^0) - H_q(X^{t}\mid X^0)\right) + H_p(X^T)\\
&=
\sum_{t=2}^T\int-\log \frac{p(x^{t-1}\mid x^t)}{q(x^{t-1}\mid x^{t},x^0)}q(x^{0:T}) dx^{0:T} + H_q(X^1\mid X^0) - H_q(X^T\mid X^0) + H_p(X^T)\\
&=
\sum_{t=2}^T\int D_{KL}(q(x^{t-1}\mid x^{t},x^0)\parallel p(x^{t-1}\mid x^t))q(x^0,x^t) dx^0dx^t + H_q(X^1\mid X^0) - H_q(X^T\mid X^0) + H_p(X^T)
\end{aligned}$

11. $\beta_t$也是可学习的（$\beta_1$固定为小常量避免过拟合），因此两个交叉熵项并非常量。

**数学过程中的近似**

- step数有限，最终的$x^T$并非严格的正态分布
- 实操中$\beta_t$并非趋于0，带来了两个后果
   - 反向过程$p(x^{t-1}\mid x^t)$并非正态分布
   - 通过轨迹求出的$p(x^0)$并非对轨迹积分的准确估计
   - Jensen不等式的近似（趋于0时，这个不等式会变成等式）
- MLP拟合正态分布的均值和方差不精确（MLP不一定是合适的结构）
- 对$x^0$的特殊处理：$p(x^0\mid x^1)=q(x^1\mid x^0)$
<a name="BORl6"></a>
### 分布的相乘
生成过程中，为了受控的生成，我们需要给模型生成的每个中间状态的分布$q(x^t)$乘一个函数$r(x^t)$，得到修改后的分布$\tilde{q}(x^t)$，用$\tilde{Z}_t$表示归一化函数：定义$\tilde{q}(x^0)$与$\tilde{q}(x^{t+1}\mid x^t)$后，以下三个公式均可从贝叶斯公式推出

- $\tilde{q}(x^t)=\frac{1}{\tilde{Z}_t}q(x^t)r(x^t)$
- $\tilde{q}(x^{t+1}\mid x^t)\propto q(x^{t+1}\mid x^t)r(x^{t+1})$
- $\tilde{q}(x^t \mid x^{t+1})\propto q(x^t \mid x^{t+1})r(x^t)$

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679625453096-cad0d1fc-1338-465b-a4c9-300ba2f67b03.png#averageHue=%23a6a6a6&clientId=u4b427259-06cb-4&from=paste&height=255&id=u9c722960&name=image.png&originHeight=391&originWidth=403&originalType=binary&ratio=1&rotation=0&showTitle=false&size=168703&status=done&style=none&taskId=u6b2bcdb1-a755-4429-a595-48242493d0c&title=&width=262.3999938964844)<br />因此，待拟合的分布p也被类似的修改：$\tilde{p}(x^t \mid x^{t+1})\propto p(x^t \mid x^{t+1})r(x^t)$。实操中比如从上图恢复中间的噪声，就会在学出来的反向过程中，每一步都对已知的图乘以一个delta函数来让它确定，对未知的图乘一个常数。
<a name="VtEVx"></a>
### 反向过程的熵
![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679714169013-c37503db-2fc0-4caf-9e50-b6e25b39306c.png#averageHue=%23faf8f6&clientId=uf63c33e3-3609-4&from=paste&height=85&id=u0cb0531b&name=image.png&originHeight=126&originWidth=725&originalType=binary&ratio=1&rotation=0&showTitle=false&size=20696&status=done&style=none&taskId=udd5c2d3f-b179-47b2-aa56-480169c54bf&title=&width=487)

- 小于等于前向过程的熵
- 大于等于前向过程的熵减去后前两step熵的增量
<a name="IXQ0P"></a>
### 一些疑问

- 什么是edge effect：看new bing的回复，应该是表示$p(x_0\mid x_1)$再往前没有数据了，不同于中间的条件概率，导致其表现与其他step的条件概率不同。
<a name="Sey5H"></a>
## 生成高清图片
> [Denoising diffusion probabilistic models](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)

<a name="Wwfe0"></a>
### 前向过程的快捷计算
利用重参数化技巧，可以计算任何一个step的图像：![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679716063541-cf1d8e31-10d3-4e65-8f09-4cea95343591.png#averageHue=%23f7f5f4&clientId=uf63c33e3-3609-4&from=paste&height=30&id=u2df95de8&name=image.png&originHeight=62&originWidth=605&originalType=binary&ratio=1&rotation=0&showTitle=false&size=8965&status=done&style=none&taskId=ud1df2986-9087-46db-b701-a44906266ac&title=&width=290)![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679716109882-1a3bbbb0-64c8-4114-96cc-fa8bb54d622b.png#averageHue=%23f6f4f2&clientId=uf63c33e3-3609-4&from=paste&height=26&id=u0499b10d&name=image.png&originHeight=53&originWidth=509&originalType=binary&ratio=1&rotation=0&showTitle=false&size=6540&status=done&style=none&taskId=u5a1f38ac-d1fb-4ca7-aaaa-3578ea3743e&title=&width=245.20001220703125)

1. $x_2 =\sqrt{\alpha_2}x_1+\sqrt{1-\alpha_2}\epsilon_1
=\sqrt{\alpha_1\alpha_2}x_0+\sqrt{\alpha_2(1-\alpha_1)}\epsilon_0+\sqrt{1-\alpha_2}\epsilon_1$
2. 为了求方差项的方差，对它做平方，得到$E[\alpha_2\epsilon_0^2-\alpha_1\alpha_2\epsilon_0^2+(1-\alpha_2)\epsilon_1^2]=1-\alpha_1\alpha_2$
<a name="NPvwB"></a>
### 优化目标的快速计算
经过文章推导，优化目标（ICML-2015中的$K=\sum_{t=1}^T\int-\log \frac{p(x^{t-1}\mid x^t)}{q(x^{t}\mid x^{t-1})} q(x^{0:T}) dx^{0:T}+H_p(X^T)$）可以写成如下形式<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679716304525-5544a77b-83db-4603-9c73-5dc392cac0e3.png#averageHue=%23f2f1f0&clientId=uf63c33e3-3609-4&from=paste&height=66&id=udd40511e&name=image.png&originHeight=128&originWidth=1400&originalType=binary&ratio=1&rotation=0&showTitle=false&size=22398&status=done&style=none&taskId=ue17d9673-9d4c-41a4-adff-948731fdfb2&title=&width=724.4000244140625)<br />这个形式比原始论文的更清晰，并且没有在这一步就把t=1（$L_0$）给去掉。<br />后验概率q可以如下计算：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679719968051-18ca88ef-a0f8-4c0b-9172-849949005223.png#averageHue=%23f9f8f7&clientId=uf63c33e3-3609-4&from=paste&height=99&id=uab73ae6b&name=image.png&originHeight=161&originWidth=1297&originalType=binary&ratio=1&rotation=0&showTitle=false&size=30884&status=done&style=none&taskId=u2078718e-1e77-45c3-8f44-c1804ff232f&title=&width=798.5999755859375)<br />因此上面的所有KL散度，都是**高斯分布**间的，不需要蒙特卡洛方法，只需要Rao-Blackwellized方法计算，有close-formed公式
<a name="IiwHV"></a>
### 学习算法

- 本文固定$\beta_t$，因此$L_T$变成一个常数，无需考虑
- 方差固定为$\sigma^2_t I$，$\sigma_t^2=\beta_t$，不学习
- 给定上述方差后，关于$L_{t-1}$可以用如下公式计算：
   1. ![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679722716115-9db44f11-afb9-450c-97a5-84275d516c2b.png#averageHue=%23fbf9f8&clientId=u09bcef6a-b0b1-4&from=paste&height=69&id=u360e2f12&name=image.png&originHeight=90&originWidth=602&originalType=binary&ratio=1&rotation=0&showTitle=false&size=10999&status=done&style=none&taskId=u42eecabd-ddbb-432a-8a3b-a30bf2ff807&title=&width=460.6000061035156)
   2. 更进一步，$x_t(x_0,\epsilon)=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$，带入得到

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679728054391-32d4da0e-1bee-4596-bbeb-e6069adbbc0c.png#averageHue=%23fbfaf9&clientId=u4684041f-d240-4&from=paste&height=204&id=ud5140d45&name=image.png&originHeight=302&originWidth=1452&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=48548&status=done&style=none&taskId=u722e7d3b-baa8-4a3c-98ae-07d39b8c662&title=&width=982.4000244140625)

- $\mu_\theta$应该学习的是![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679728266870-63e9f1e3-75f4-4f43-9d29-a276ea1a67f2.png#averageHue=%23f7f6f5&clientId=u4684041f-d240-4&from=paste&height=59&id=u8a04f34c&name=image.png&originHeight=90&originWidth=411&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=14207&status=done&style=none&taskId=u64c9e6d4-fea6-4cda-b026-bf907acee20&title=&width=267.8000183105469)，
   - 从而我们拆解![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679728386334-d8de5d3c-6942-49fc-b70d-1e805bbad839.png#averageHue=%23f9f7f7&clientId=u4684041f-d240-4&from=paste&height=65&id=u25f6b9e6&name=image.png&originHeight=111&originWidth=1321&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=20286&status=done&style=none&taskId=u26a3295b-6dd3-4a9a-bcc8-78be7940e8c&title=&width=776.4000244140625)
   - 生成阶段，采用公式![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679728871856-47df41c6-e674-428d-9ac8-e4f6c09a855f.png#averageHue=%23f7f5f4&clientId=u4684041f-d240-4&from=paste&height=57&id=u12f50400&name=image.png&originHeight=77&originWidth=968&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=17292&status=done&style=none&taskId=u43a40c93-be56-4a32-96e3-724477ee250&title=&width=710.4000244140625)

![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679728626577-66ee24a8-5c96-4c15-9ac7-3aac1a862f14.png#averageHue=%23f3f2f2&clientId=u4684041f-d240-4&from=paste&height=241&id=u01bb0181&name=image.png&originHeight=377&originWidth=1526&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=90703&status=done&style=none&taskId=u9a06a69d-8260-4fd0-bbb2-6f6e1dbce5b&title=&width=976.4000244140625)<br />**前面所有复杂的数学公式，得到的loss可以简单定义为，实际噪声与预测噪声的平方差**<br />**前面公式的作用在于生成阶段**$x_{t-1}$**的计算**<br />此外，前面推导中不同step的weight应该是不同的：![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679896087102-b19248b8-a1d7-4574-9b63-b41e8ef9b01c.png#averageHue=%23f8f7f6&clientId=u4684041f-d240-4&from=paste&height=43&id=u8ee0c36e&name=image.png&originHeight=104&originWidth=903&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=18489&status=done&style=none&taskId=u26ef52fb-6110-4569-bdf4-3c2632ed723&title=&width=372.4000244140625)，这里把这个weight去掉了实际相当于强调了与理论bound不同的某些step（对小的t取了偏小的weight），这个side-effect实际更有利于模型训练，因为对更大噪声的去噪更困难，需要网络更多去考虑。
<a name="L7EZ9"></a>
### 最后一步$p(x_0\mid x_1)$的处理
图像数据被从[0,255]normalize到[-1,1]，为了确定discrete的像素值，文章会对最后学习到的$p(x_0\mid x_1)$的连续条件分布取一个像素点对应区间的积分，得到离散的像素值（i表示数据的下标，即channel、h、w）：<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679895075823-9d4f4606-b43a-40bf-bef3-463fbb6ab75d.png#averageHue=%23fbfaf9&clientId=u4684041f-d240-4&from=paste&height=165&id=u61c11829&name=image.png&originHeight=250&originWidth=1103&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=35405&status=done&style=none&taskId=uc75eccaa-c000-4896-a401-30b2422b852&title=&width=726.4000244140625)<br />注：与其他step的采样的区别是，最后这一步对方差的处理没有随机性，而是noiseless的（直接取确定的积分）
<a name="UVNSi"></a>
### 其他细节（实验）
$\beta_t$：从$10^{-4}$线性增长到$0.02$，文章解释这样会尽可能减小$L_T$，即前向结果与正态分布的一致性。<br />backbone：U-Net，不同t的参数共享，t以Transformer sinusoidal position embedding的形式喂入网络。<br />**Ablations**<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679898439544-a4954de7-63be-4ed1-99ee-60247c46b607.png#averageHue=%23f0eeed&clientId=u4684041f-d240-4&from=paste&height=219&id=u2cc05bea&name=image.png&originHeight=304&originWidth=486&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=42919&status=done&style=none&taskId=uf98ff680-c681-485f-aff4-b3d50a1f42f&title=&width=349.8000183105469)

- 模型直接predict$\mu$而非$\epsilon$：不能使用简化了的L，必须使用原始的weighted form，否则效果很差
- 模型使用学习出来的方差矩阵（对角阵）：训练不稳定

**通过共享不同step的图片，可以生成不同相似度的图：**<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679899492835-f5463b09-399b-41ab-8c82-c9eb90a4825e.png#averageHue=%237f7660&clientId=u4684041f-d240-4&from=paste&height=278&id=uf948e24d&name=image.png&originHeight=348&originWidth=1246&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=755329&status=done&style=none&taskId=uf3a42622-9cd6-4cde-a6ff-049d402b495&title=&width=996.8)<br />**Diffusion模型适合于图像插值：**<br />![image.png](https://cdn.nlark.com/yuque/0/2023/png/2787610/1679899834706-5b7fa77c-ff54-4e04-8a7f-6682ecf1613b.png#averageHue=%23c0b2a9&clientId=u4684041f-d240-4&from=paste&height=182&id=u1fb68210&name=image.png&originHeight=356&originWidth=1530&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=611802&status=done&style=none&taskId=ud7dff6ff-4f47-4115-b74e-8e4da8d0239&title=&width=782.4000244140625)

