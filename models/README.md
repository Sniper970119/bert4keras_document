# Sniper

Model主要存放一些模型。比如Trm、Bert、T5等。

## class Transformer()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L13)

    class Transformer(object)

模型基类。所有Transformer based（Bert以及各种变种、T5等）的模型的基类。

    def __init__(
        self,
        vocab_size,  # 词表大小
        hidden_size,  # 编码维度
        num_hidden_layers,  # Transformer总层数
        num_attention_heads,  # Attention的头数
        intermediate_size,  # FeedForward的隐层维度
        hidden_act,  # FeedForward隐层的激活函数
        dropout_rate=None,  # Dropout比例
        embedding_size=None,  # 是否指定embedding_size
        attention_head_size=None,  # Attention中V的head_size
        attention_key_size=None,  # Attention中Q,K的head_size
        sequence_length=None,  # 是否固定序列长度
        keep_tokens=None,  # 要保留的词ID列表
        compound_tokens=None,  # 扩展Embedding
        residual_attention_scores=False,  # Attention矩阵加残差
        ignore_invalid_weights=False,  # 允许跳过不存在的权重
        layers=None,  # 外部传入的Keras层
        prefix=None,  # 层名前缀
        name=None,  # 模型名称
        **kwargs
    )

大部分参数代码注释比较完善，需要格外说明的：

`hierarchical`默认为None，为True时为使用超长编码(利用层次分解，将bert（Transformer）的最长512的序列长度扩充为512*512，会损失一定精度，
但是微调后可以使用很小的代价恢复性能) [苏神博客](https://kexue.fm/archives/7947 )


`residual_attention_scores`是否使用残差Attention矩阵。残差Attention矩阵，给每个Attention矩阵加上前上一层的Attention矩阵，
来源[RealFormer](https://arxiv.org/abs/2012.11747 ),目前的实现可能还相对粗糙，欠缺通用性。

`ignore_invalid_weights` 为是否允许跳过名字不匹配的权重。默认为False，为True时，遇到名字不匹配的层名字时，
会输出一个报错信息，但是程序并不会终止，改层的权重会随机初始化。

    def build(
            self,
            attention_caches=None,
            layer_norm_cond=None,
            layer_norm_cond_hidden_size=None,
            layer_norm_cond_hidden_act=None,
            additional_input_layers=None,
            **kwargs
        ):

`attention_caches` 为Attention的K,V的缓存序列字典，格式为{Attention层名: [K缓存, V缓存]}；

`layer_norm_*`系列参数：实现`Conditional Layer Normalization`时使用，用来实现以“固定长度向量”为条件的条件Bert。该方法通过在LN层加入一个方向的扰动，从而可以在一个模型中完成多个类似的任务，
比如在一个模型中生成积极的文本和消极的文本、在一个模型中进行短短文本匹配，短长文本匹配等。详见[苏神博客](https://kexue.fm/archives/7124 )

`additional_input_layers`为除Bert原生输入外其余的输入项。通过`self.set_inputs()`来添加到模型中。

    def call(self, inputs):
        """定义模型的执行流程
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs

call方法可以看出来，整体来说，是embedding、main layers（Transformer）、final layers（dense）。

    def set_inputs(self, inputs, additional_input_layers=None):
        """设置input和inputs属性
        """
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        inputs = inputs[:]
        if additional_input_layers is not None:
            if not isinstance(additional_input_layers, list):
                additional_input_layers = [additional_input_layers]
            inputs.extend(additional_input_layers)

        self.inputs = inputs
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]

set_inputs方法可以看出来如何添加的`additional_input_layers`，同时处理input参数。
（input/inputs区分一下，我研究半天这是干嘛的，后来发现不一样）。

    def load_embeddings(self, embeddings):
        """处理Embedding层权重
        """
        embeddings = embeddings.astype(K.floatx())  # 防止np.average报错

        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                if isinstance(item, list):
                    item = (item, [1] * len(item))
                ext_embeddings.append(
                    np.average(embeddings[item[0]], 0, item[1])
                )
            embeddings = np.concatenate([embeddings, ext_embeddings], 0)

        return embeddings

load_embedding分别对应的缩小embedding（keep_token）和扩大embedding(compound_token)两种情况。

前者用于不需要这么多token（比如bert4keras默认的精简方式详见[参数simplified](https://github.com/Sniper970119/bert4keras_document/tree/master/tokenizers#def-load_vocab )）
，只需要将embedding对应部分截取出来就行。
后者对应需要更多的token，直接在embedding中添加新的行（axis=0）就行了。

### class LM_Mask()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L349)

    class LM_Mask(object)

定义下三角Attention Mask（语言模型用）

    def lm_mask(s):
        seq_len = K.shape(s)[1]
        idxs = K.arange(0, seq_len)
        mask = idxs[None, :] <= idxs[:, None]
        mask = K.cast(mask, K.floatx())
        return -(1 - mask[None, None]) * 1e12

这里就是计算一个下三角矩阵，通过s（s -> [batch_size,segment_ids]）计算mask矩阵。用于进行masked language model。

使用只需要在`build_transformer_model`中添加`application='lm'`即可。

这里`mask = idxs[None, :] <= idxs[:, None]`添加两个None维度是为了便于idx的错位比较

不过这里我仍然有一个未解之谜，就是为什么要对mask后的矩阵添加两个维度，问过苏神，说是multi-head-attention需要，
但是我看了multi-head-attention部分的源码还是没太明白，我太菜了不敢继续问苏神，等我继续摸索摸索。

example:

    model = build_transformer_model(
    config_path = config_path,
    checkpoint_path = checkpoint_path,
    model='bert',
    application='lm',
    )
    

### class UniLM_Mask()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L374)

    class UniLM_Mask(object)

定义UniLM的Attention Mask（Seq2Seq模型用）
[UniLM](https://arxiv.org/abs/1905.03197 )[苏神博客](https://kexue.fm/archives/6933 )

     def unilm_mask(s):
        idxs = K.cumsum(s, axis=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        mask = K.cast(mask, K.floatx())
        return -(1 - mask[:, None]) * 1e12

这里就是通过s（s -> [batch_size,segment_ids]）的segment_ids为1的地方进行下三角矩阵mask，用以完成UniLM的Seq2Seq任务。

使用只需要在`build_transformer_model`中添加`application='unilm'`即可。

`idxs = K.cumsum(s, axis=1)` 对列进行求和（eg：[0,0,1,1,1]则返回[0,0,1,2,3]）。

这里`idxs[:, None, :] <= idxs[:, :, None]`添加两个None维度是为了便于idx的错位比较

这里依然由上面的问题，上面是对mask最前面添加了两个维度，而这里对第二维添加了一个维度，不太清楚为啥。。果然还是我太菜了。
回头想明白回来填坑。

example:

    model = build_transformer_model(
    config_path = config_path,
    checkpoint_path = checkpoint_path,
    model='bert',
    application='unilm',
    )
    




















### def strQ2B()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L13)

    def strQ2B(ustring)

其中：

|参数| 说明|
|:-----  |-----|
|ustring|全角字符串:str|