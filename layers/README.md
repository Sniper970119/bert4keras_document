# Sniper

这个文件主要为定义自定义层，在`model.py`中使用。

## class Layer()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L31 )

    class Layer(keras.layers.Layer)
    
这里苏神做了版本统一与框架自定义。

对低于keras2.3（tf1）的版本，添加层中层的概念（也就是一层中可以有多层）。

> 相比原生keras，通过在对象中[添加_layers列表](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L43 )，将“子层”添加到列表中从而做到支持层中层。

同时对于keras版本低于2.2.5重新定义Model，整合fit和git_generator()

另外单独一说，keras的supports_masking默认为False，bert4keras全部为True，即支持单独mask。

[keras中supports_masking为False](https://github.com/keras-team/keras/blob/keras-2/keras/engine/topology.py#L249 )

[bert4keras中supports_masking默认为True](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L39 )

## class GlobalAveragePooling1D()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L122 )

    class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D)
    
重新定义GlobalAveragePooling1D，支持序列长度为None。

## class GlobalMaxPooling1D()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L136 )

    class GlobalMaxPooling1D(keras.layers.GlobalMaxPooling1D)
    
重新定义GlobalMaxPooling1D，支持mask。

通过 backend的[def sequence_masking()](https://github.com/Sniper970119/bert4keras_document/tree/master/backend#def-sequence_masking ) 来进行mask。

## class Embedding()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L156 )

    class Embedding(keras.layers.Embedding)

拓展Embedding层

相比原生的Embedding，主要适配了T5以及其类似模型，保证第一个token不能被mask。

## class BiasAdd()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L192 )

    class BiasAdd(Layer)

在这一层加上一个可训练的偏置，用于`model.py`中的模型构建。

## class Concatenate1D()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L210 )

    class Concatenate1D(Layer)

1维序列拼接层

说明：本来该功能可以直接通过Concatenate层来实现，无奈Keras自带的Concatenate层的compute_mask写得不合理，导致一个
mask的序列与一个不带mask的序列拼接会报错，因此干脆自己重写一个好了。

对比tensorflow中的[代码](https://github.com/tensorflow/tensorflow/blob/r2.4/tensorflow/python/keras/layers/merge.py#L542 ),
可以发现bert4keras在compute_mask这里并不是像tensorflow中最直接append到masks中，而是如果mask为空，则初始化一个一样的mask矩阵。

tensorflow：

      if mask_i is None:
        # Input is unmasked. Append all 1s to masks,
        masks.append(array_ops.ones_like(input_i, dtype='bool'))
      elif K.ndim(mask_i) < K.ndim(input_i):
        # Mask is smaller than the input, expand it
        masks.append(array_ops.expand_dims(mask_i, axis=-1))
      else:
        masks.append(mask_i)

bert4keras：

      if mask is not None:
          masks = []
          for i, m in enumerate(mask):
              if m is None:
                  m = K.ones_like(inputs[i][..., 0], dtype='bool')
              masks.append(m)
          return K.concatenate(masks, axis=1)


## class MultiHeadAttention()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L122 )

    class MultiHeadAttention(Layer)

多头注意力机制

其中：
|参数| 说明|
|:-----  |-----|
|heads  |头的个数:int|
|head_size  |头的大小:int|
|out_dim  |输出维度:int|
|key_size  |论文中的alpha:int|
|use_bias  |是否使用偏置:bool|
|attention_scale  |是否使用小规模参数:bool|
|return_attention_scores  |是否返回注意力分数:bool|
|kernel_initializer  |初始化方式:str|


其中，

`heads` 就是多头注意力的头个数，比如bert-base为12。

`head_size`就是每个头的大小，比如bert-base为64。

`out_dim` 默认为None。 为None时就是heads * head_size ，不为None则后面跟一个dense层处理到out_dim这个维度。

`key_size`默认为None。为None时就是head_size，不一样时，Q和K会被初始化为key_size*heads。至于这个参数非默认情况的用途。。我没看出来。。等见过之后回来补充。

`use_bias`是否使用偏置（传入的QKV首先会经过一次线性变换后然后计算attention，这里的偏置指的是这个线性变化的，默认为True）。

`attention_scale` 返回的注意力参数规模，如果为True则开方（sqrt）后返回。

`return_attention_scores`是否返回注意力分数。

`kernel_initializer` 参数初始化方式，默认为`glorot_uniform`。

在实际计算过程中，q_mask并没有使用（但是参与了全程的参数传递），最终的mask是通过计算完attention通过v_mask的值mask掉attention的输出。

这里通过一行代码

    qkv_inputs = [qw, kw, vw] + inputs[3:]

来适配将相对位置嵌入到注意力机制中（[NEZHA中使用](https://github.com/Sniper970119/bert4keras_document/tree/master/models#class-NEZHA )）。




## class LayerNormalization()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L399 )

    class LayerNormalization(Layer)

(Conditional) Layer Normalization

hidden_*系列参数仅为有条件输入时(conditional=True)使用

其中：
|参数| 说明|
|:-----  |-----|
|center  |是否使用beta:bool|
|scale  |是否使用gamma:bool|
|epsilon  |一个极小值:float|
|conditional  |是否为Conditional Layer Normalization:bool|
|hidden_units  |隐藏层个数:int|
|hidden_activation  |隐藏层激活方法:str|
|hidden_initializer  |初始化方式:str|

其中，

`center` 是否使用beta。

`scale` 是否使用gamma。

`epsilon` 默认为1e-12。 用于计算标准差时的极小值。

`conditional`False。为True时为使用Conditional Layer Normalization，该方法通过在LN层加入一个方向的扰动，从而可以在一个模型中完成多个类似的任务，
比如在一个模型中生成积极的文本和消极的文本、在一个模型中进行短短文本匹配，短长文本匹配等。详见[苏神博客](https://kexue.fm/archives/7124 )

`hidden_units`隐藏层个数，用于控制在Conditional Layer Normalization的embedding大小。

`hidden_activation` 隐藏层激活方法，默认为`linear`。

`hidden_initializer` 参数初始化方式，默认为`glorot_uniform`。


## class PositionEmbedding()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L524 )

    class PositionEmbedding(Layer)

定义可训练的位置Embedding(比如Bert的position embedding)

    def __init__(
            self,
            input_dim,
            output_dim,
            merge_mode='add',
            hierarchical=None,
            embeddings_initializer='zeros',
            custom_position_ids=False,
            **kwargs
        )

`hierarchical`默认为None，为True时为使用超长编码(利用层次分解，将bert的最长512的序列长度扩充为512*512，会损失一定精度，
但是微调后可以使用很小的代价恢复性能) [苏神博客](https://kexue.fm/archives/7947 )





## class SinusoidalPositionEmbedding()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L612 )

    class SinusoidalPositionEmbedding(Layer)

定义Sin-Cos位置Embedding(比如transformer的position embedding)


## class RelativePositionEmbedding()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L672 )

    class RelativePositionEmbedding(Layer)

计算相对位置编码。(比如NEZHA的position embedding)[代码](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1085),
[文档](https://github.com/Sniper970119/bert4keras_document/tree/master/models#class-NEZHA )


## class RelativePositionEmbeddingT5()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L727 )

    class RelativePositionEmbeddingT5(Layer)

Google T5的相对位置编码  https://arxiv.org/abs/1910.10683


## class FeedForward()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L786 )

    class FeedForward(Layer):

FeedForward层

        def __init__(
        self,
        units,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    )


如果activation不是一个list，那么它就是两个Dense层的叠加；

如果activation是一个list，那么第一个Dense层将会被替换成门控线性单元（Gated Linear Unit）。

[参考论文(T5.1.1的论文，通过使用GLU来增强FFN的效果。)](https://arxiv.org/abs/2002.05202 )
[苏神博客](https://kexue.fm/archives/7867#T5.1.1 )


## class ConditionalRandomField()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L850 )

    class ConditionalRandomField(Layer)

纯Keras实现CRF层，CRF层本质上是一个带训练参数的loss计算层。

条件随机场，整个框架内暂时没有被（调）用过。


## class MaximumEntropyMarkovModel()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L971 )

    class MaximumEntropyMarkovModel(Layer):


（双向）最大熵隐马尔可夫模型，作用和用法都类似CRF，但是比CRF更快更简单。

同样，整个框架内暂时没有被（调）用过。


## class Loss()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/layers.py#L1158 )

    class Loss(Layer)

特殊的层，用来定义复杂loss。

通过引入一个层，来实现复杂的loss计算。在层内部通过将计算后的loss添加到self.add_loss从而定义这个层的loss。

    def call(self, inputs, mask=None):
        loss = self.compute_loss(inputs, mask)
        self.add_loss(loss, inputs=inputs)
        if self.output_axis is None:
            return inputs
        elif isinstance(self.output_axis, list):
            return [inputs[i] for i in self.output_axis]
        else:
            return inputs[self.output_axis]

    def compute_loss(self, inputs, mask=None):
        raise NotImplementedError

通过`self.add_loss(loss, inputs=inputs)`定义改层loss，并且返回`inputs`，因此并不改变整个网络的输出。

而add_loss 则是`class Layer(object):`中的一个方法，定义如下：

    def add_loss(self, losses, inputs=None):
        """Adds losses to the layer.

        The loss may potentially be conditional on some inputs tensors,
        for instance activity losses are conditional on the layer's inputs.

        # Arguments
            losses: loss tensor or list of loss tensors
                to add to the layer.
            inputs: input tensor or list of inputs tensors to mark
                the losses as conditional on these inputs.
                If None is passed, the loss is assumed unconditional
                (e.g. L2 weight regularization, which only depends
                on the layer's weights variables, not on any inputs tensors).
        """

这样添加了一个loss层来计算loss作为整个模型的loss。实在是。。太妙了，牛b。

example:

    class CrossEntropy(Loss):
        """交叉熵作为loss，并mask掉padding部分
        """
    
        def compute_loss(self, inputs, mask=None):
            y_true, y_pred = inputs
            if mask[1] is None:
                y_mask = 1.0
            else:
                y_mask = K.cast(mask[1], K.floatx())[:, 1:]
            y_true = y_true[:, 1:]  # 目标token_ids
            y_pred = y_pred[:, :-1]  # 预测序列，错开一位
            loss = K.sparse_categorical_crossentropy(y_true, y_pred)
            loss = K.sum(loss * y_mask) / K.sum(y_mask)
            return loss

    model = build_transformer_model(
        config_path,
        checkpoint_path,
        application='lm',
    )
    output = CrossEntropy(1)([model.inputs[0], model.outputs[0]])

这里自定义了一个可以mask的损失函数（MLM的），然后下面通过将Bert的输入和输出一同送入计算loss，并返回 `model.outputs[0]` 也就是索引1.

这样就可以“透明”的添加一个自定义loss，而不改变模型输出。