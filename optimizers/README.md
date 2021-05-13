# Sniper

`optimizers`为各种优化器所在的文件

代码中所有的v1 v2为苏神自己的迭代版本号，实际上调用调用没有v的就行，有变量指向这些方法。

由于本文是根据源代码写的，因此保留这些版本号。

## class Adam()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L13)

    class Adam(keras.optimizers.Optimizer)
    
重新定义Adam优化器，便于派生出新的优化器（tensorflow的optimizer_v2类）

检查了一下tensorflow中对于Adam的[源码](https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/optimizer_v2/adam.py#L107 )，
没发现什么不同却又处处不同。读了一下代码，大概猜测就如苏神说的一样，“重新定义Adam优化器，便于派生出新的优化器”。
将tensorflow的代码提取，并去掉AmsGrad的支持，简化了一下。

## class AdaFactorBase

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L92)


    class AdaFactorBase(keras.optimizers.Optimizer)
    
AdaFactor优化器（基类）

[论文链接](https://arxiv.org/abs/1804.04235 )、[参考实现](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py )

但是这个优化器本身是Google提出来的，它具有自适应学习率，但是比RMSProp还要省显存，并针对性解决了一些Adam的缺陷。

这么好，又省显存又解决缺陷，那为啥不用呢？

> 需要提醒的是，用AdaFactor的时候，batch_size最好大一些，因为本身低秩分解会带来误差，而如果batch_size过小，
那么梯度估算本身也带来较大的误差，两者叠加优化过程可能还不收敛。对于预训练模型来说，batch_size通常还是很大的，
所以现在不少预训练模型开始用AdaFactor优化器了；对于普通的下游任务来说，AdaFactor也可以尝试，但可能需要多炼炼丹，
才能搞出优于无脑Adam的效果。对了，还要提醒一下，用AdaFactor的时候，学习率要设大一点，大概是10的-3级别为好，
哪怕是finetune阶段也是如此。

[详见苏神博客](https://kexue.fm/archives/7302 )


## class AdaFactorV1()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L171)


    class AdaFactorV1(AdaFactorBase)
    

AdaFactor优化器（纯Keras版）

[论文链接](https://arxiv.org/abs/1804.04235 )、[参考实现](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py )

tensorflow1.x 版本的AdaFactor（keras）

详见 [ class AdaFactorBase](https://github.com/Sniper970119/bert4keras_document/tree/master/optimizers#class-AdaFactorV2 )

## class AdaFactorV2()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L236)


    class AdaFactorV1(AdaFactorBase)
    

AdaFactor优化器（tf.keras版）

[论文链接](https://arxiv.org/abs/1804.04235 )、[参考实现](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/optimize.py )

tensorflow2.x 版本的AdaFactor（tf.keras）

详见 [ class AdaFactorBase](https://github.com/Sniper970119/bert4keras_document/tree/master/optimizers#class-AdaFactorV2 )

### def extend_with_weight_decay()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L325)

    def extend_with_weight_decay(BaseOptimizer)
    
返回加入权重衰减的新优化器，已弃用


### def extend_with_weight_decay_v2()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L367)

    def extend_with_weight_decay_v2(BaseOptimizer)
    
返回加入权重衰减的新优化器。

L2正则化的目的就是为了让权重衰减到更小的值，在一定程度上减少模型过拟合的问题，所以权重衰减也叫L2正则化。

更小的权值w，从某种意义上说，表示网络的复杂度更低，对数据的拟合更好（这个法则也叫做奥卡姆剃刀），而在实际应用中，也验证了这一点，L2正则化的效果往往好于未经正则化的效果。


### def extend_with_layer_adaptation()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L407)

    def extend_with_layer_adaptation(BaseOptimizer)
    
返回加入层自适应学习率的新优化器，已弃用


### def extend_with_layer_adaptation_v2()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L462)

    def extend_with_layer_adaptation_v2(BaseOptimizer)
    
返回加入层自适应学习率的新优化器。


带有层自适应学习率的优化器,用每一层参数的模长来校正当前参数的学习率。

LAMB（Layer-wise Adaptive Moments optimizer for Batching training）优化器，将Bert的训练时间从3天减少到了76分钟。

> 为了训练BERT, Devlin等人首先使用序列长度为128的900k迭代训练模型，然后在最后的100k迭代中转换为512的序列长度。这导致了在16个TPUv3芯片上大约需要3天的训练时间。（baseline）
>
>将训练优化器更改为LAMB之后，保持与基线相同的训练过程，使用与基线相同数量的epochs运行，
>但批大小（batch size）从512扩展到32K（选择32K大小（序列长度512）主要是由于TPU Pod的内存限制）。
>通过使用LAMB优化器，能够在批大小（batch size）为32k的15625次迭代（序列长度为128的14063次迭代和序列长度为512的1562次迭代）
>中获得91.460的F1分数。
>对于批大小（batch size）为32K，本文将BERT训练时间从3天缩短到约100分钟。

该论文来自Google，被收录进ICLR2020。[论文](https://arxiv.org/abs/1904.00962 )


### def extend_with_piecewise_linear_lr()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L515)

    def extend_with_piecewise_linear_lr(BaseOptimizer)
    

带有分段线性学习率的优化器,其中schedule是形如{1000: 1, 2000: 0.1}的字典，,表示0～1000步内学习率线性地从零增加到100%，然后
1000～2000步内线性地降到10%，2000步以后保持10%。

已弃用


### def extend_with_piecewise_linear_lr_v2()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L557)

    def extend_with_piecewise_linear_lr_v2(BaseOptimizer)
    
带有分段线性学习率的优化器,其中schedule是形如{1000: 1, 2000: 0.1}的字典，,表示0～1000步内学习率线性地从零增加到100%，然后
1000～2000步内线性地降到10%，2000步以后保持10%。


### def extend_with_gradient_accumulation()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L587)

    def extend_with_gradient_accumulation(BaseOptimizer)
    

带有梯度累积的优化器。

已弃用


### def extend_with_gradient_accumulation_v2()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L648)

    def extend_with_gradient_accumulation_v2(BaseOptimizer)
    
带有梯度累积的优化器。

梯度累计：就是类似于torch不清空梯度，累计几次再进行一次反向传播、清空梯度。从而“放大”batch size。同时学习率也要对应增大。



### def extend_with_lookahead()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L702)

    def extend_with_lookahead(BaseOptimizer)
    

带有look ahead的优化器。

其中：
|参数| 说明|
|:-----  |-----|
|steps_per_slow_update  |论文中的k:int|
|slow_step_size  |论文中的alpha:float|

已弃用


### def extend_with_lookahead_v2()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L752)

    def extend_with_lookahead_v2(BaseOptimizer)
    
带有look ahead的优化器。

其中：
|参数| 说明|
|:-----  |-----|
|steps_per_slow_update  |论文中的k:int|
|slow_step_size  |论文中的alpha:float|

look ahead：通过一个内循环优化器（它可以是任何优化器，Adam、SGD）提前循环k次权重，称作Fast Weights。
然后由外层循环将内层计算的k次权重通过权重滑动平均（EMA）计算新的方向，得到Slow Weights。

最终模型使用的参数是Slow Weights，Fast Weights相当于做了一系列实验“look ahead”。

[论文](https://arxiv.org/abs/1907.08610 )


### def extend_with_lazy_optimization()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L799)

    def extend_with_lazy_optimization(BaseOptimizer)
    

带有懒惰更新的优化器。使得部分权重（尤其是embedding）只有在梯度不等于0时更新。

已弃用


### def extend_with_lazy_optimization_v2()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L853)

    def extend_with_lazy_optimization_v2(BaseOptimizer)
    
带有懒惰更新的优化器。使得部分权重（尤其是embedding）只有在梯度不等于0时更新。


### def extend_with_exponential_moving_average()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L903)

    def extend_with_exponential_moving_average(BaseOptimizer)
    

带EMA（权重滑动平均，Exponential Moving Average）的优化器。

已弃用


### def extend_with_exponential_moving_average_v2()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L956)

    def extend_with_exponential_moving_average_v2(BaseOptimizer)
    
带EMA（权重滑动平均，Exponential Moving Average）的优化器。

可以用来估计变量的局部值，使得变量的更新与一段时间内的历史值有关。

维护一个shadow weight，这个shadow weight为前n次计算的weight的平均值。

只能在测试阶段使用，训练阶段依然要使用真实的weight。

[苏神博客](https://kexue.fm/archives/6575#%E6%9D%83%E9%87%8D%E6%BB%91%E5%8A%A8%E5%B9%B3%E5%9D%87 )

example:

    from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
    AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
    optimizer = AdamEMA(learing_rate, ema_momentum=0.9999)

### def extend_with_parameter_wise_lr()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L1021)

    def extend_with_parameter_wise_lr(BaseOptimizer)
    

带分参数学习率的优化器，主要场景就是给每层甚至每个参数设置不同的学习率。

已弃用


### def extend_with_parameter_wise_lr_v2()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py#L1067)

    def extend_with_parameter_wise_lr_v2(BaseOptimizer)
    
带分参数学习率的优化器，主要场景就是给每层甚至每个参数设置不同的学习率。

其中schedule是形如{name1: 2, name2: 0.1}的字典，其实name1、name2是字符串，表示变量名包含name1的
参数学习率乘以2，变量名包含name2的参数学习率要乘以0.1。

