# Sniper

Model主要存放一些模型。比如Trm、Bert、T5等。
* * *
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
来源RealFormer[论文](https://arxiv.org/abs/2012.11747 ),目前的实现可能还相对粗糙，欠缺通用性。

`ignore_invalid_weights` 为是否允许跳过名字不匹配的权重。默认为False，为True时，遇到名字不匹配的层名字时， 会输出一个报错信息，但是程序并不会终止，改层的权重会随机初始化。

### def build(self):

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

`layer_norm_*`系列参数：实现`Conditional Layer Normalization`
时使用，用来实现以“固定长度向量”为条件的条件Bert。该方法通过在LN层加入一个方向的扰动，从而可以在一个模型中完成多个类似的任务，
比如在一个模型中生成积极的文本和消极的文本、在一个模型中进行短短文本匹配，短长文本匹配等。详见[苏神博客](https://kexue.fm/archives/7124 )

`additional_input_layers`为除Bert原生输入外其余的输入项。通过`self.set_inputs()`来添加到模型中。

### def call(self):

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

### def set_inputs(self):

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

set_inputs方法可以看出来如何添加的`additional_input_layers`，同时处理input参数。 （input/inputs区分一下，我研究半天这是干嘛的，后来发现不一样，如果你观察过`bert4keras`
的模型你就会发现有input和inputs两个变量）。

### def load_embeddings(self):

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

前者用于不需要这么多token（比如bert4keras默认的精简方式详见[参数simplified](https://github.com/Sniper970119/bert4keras_document/tree/master/tokenizers#def-load_vocab )
） ，只需要将embedding对应部分截取出来就行。 后者对应需要更多的token，直接在embedding中添加新的行（axis=0）就行了。


### def compute_attention_bias(self)

    def compute_attention_bias(self, inputs=None):
        """定义每一层的Attention Bias
        """
        return self.attention_bias

这个方法主要是计算attention的mask（或者bias）比如在[LM_MASK](https://github.com/Sniper970119/bert4keras_document/tree/master/models#class-LM_Mask )以及[UniLM_Mask](https://github.com/Sniper970119/bert4keras_document/tree/master/models#class-UniLM_Mask )
中复写的`compute_attention_bias`，用于相关用途（在attention阶段添加mask[比如LM中的随机Mask]或bias[比如NEZHA在attention中添加相对位置编码]）。



* * *
## class LM_Mask()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L349)

    class LM_Mask(object)

定义下三角Attention Mask（语言模型用）

        def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        if self.attention_bias is None:

            def lm_mask(s):
                seq_len = K.shape(s)[1]
                idxs = K.arange(0, seq_len)
                mask = idxs[None, :] <= idxs[:, None]
                mask = K.cast(mask, K.floatx())
                return -(1 - mask[None, None]) * 1e12

            self.attention_bias = self.apply(
                inputs=self.inputs[0],
                layer=Lambda,
                function=lm_mask,
                name='Attention-LM-Mask'
            )

        return self.attention_bias

这里就是计算一个下三角矩阵，通过s（s -> [batch_size,token_ids]）计算mask矩阵。用于进行语言模型的训练（其实就是GPT-2的思路）。

使用只需要在`build_transformer_model`中添加`application='lm'`即可。

这里`mask = idxs[None, :] <= idxs[:, None]`添加两个None维度是为了便于idx的错位比较

最后输出[1,1,token_len,token_len]，最后两个token_len为mask矩阵。用于拼接在MultiHeadAttention的输入中。

详见[multi-head-attention](https://github.com/Sniper970119/bert4keras_document/tree/master/layers#class-multiheadattention )

example:

    model = build_transformer_model(
    config_path = config_path,
    checkpoint_path = checkpoint_path,
    model='bert',
    application='lm',
    )
* * *
### class UniLM_Mask()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L374)

    class UniLM_Mask(object)

定义UniLM的Attention Mask（Seq2Seq模型用）
[UniLM](https://arxiv.org/abs/1905.03197 )[苏神博客](https://kexue.fm/archives/6933 )

         def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        if self.attention_bias is None:

            def lm_mask(s):
                seq_len = K.shape(s)[1]
                idxs = K.arange(0, seq_len)
                mask = idxs[None, :] <= idxs[:, None]
                mask = K.cast(mask, K.floatx())
                return -(1 - mask[None, None]) * 1e12

            self.attention_bias = self.apply(
                inputs=self.inputs[0],
                layer=Lambda,
                function=lm_mask,
                name='Attention-LM-Mask'
            )

        return self.attention_bias

这里就是通过s（s -> [batch_size,segment_ids]）的segment_ids为1的地方进行下三角矩阵mask，用以完成UniLM的Seq2Seq任务。

使用只需要在`build_transformer_model`中添加`application='unilm'`即可。

`idxs = K.cumsum(s, axis=1)` 对列进行求和（eg：[0,0,1,1,1]则返回[0,0,1,2,3]）。

这里`idxs[:, None, :] <= idxs[:, :, None]`添加两个None维度是为了便于idx的错位比较

最后输出[batch size,1,token_len,token_len]，最后两个token_len为mask矩阵。用于拼接在MultiHeadAttention的输入中。

详见[multi-head-attention](https://github.com/Sniper970119/bert4keras_document/tree/master/layers#class-multiheadattention )

example:

    model = build_transformer_model(
    config_path = config_path,
    checkpoint_path = checkpoint_path,
    model='bert',
    application='unilm',
    )

* * *

## class BERT()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#400)

    class BERT(Transformer)

Bert类，继承了`Transformer`类

    def __init__(
        self,
        max_position,  # 序列最大长度
        segment_vocab_size=2,  # segment总数目
        with_pool=False,  # 是否包含Pool部分
        with_nsp=False,  # 是否包含NSP部分
        with_mlm=False,  # 是否包含MLM部分
        hierarchical_position=None,  # 是否层次分解位置编码
        custom_position_ids=False,  # 是否自行传入位置id
        shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
        **kwargs  # 其余参数
    )

我们可以发现，苏神在这里还支持了多segment_idx(原生bert仅支持两句话，也就是segment_vocab_size=2)。

当然多segment是有代价的，就是原bert的segment需要被弃用，需要在代码的`def load_weights_from_checkpoint`（Transformer类的类方法）中将`Embedding-Segment`移除`mapping`（`mapping.pop('Embedding-Segment'）`)从而不再初始化这一部分权重。

`with_pool`就是最后CLS的768维、`with_nsp`就是是否进行NSP任务（当进行NSP任务时，`with_pool`必须为True。因为nsp需要CLS向量。当然，这一步代码可以自动处理），

        if self.with_nsp and not self.with_pool:
            self.with_pool = True

`with_mlm`也是是否进行MLM任务。

`hierarchical_position`对应的层次编码，以让bert可以处理512*512长度的文本[苏神博客](https://kexue.fm/archives/7947 )。

### def apply_embeddings(self):

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L454 )

这个方法为BERT的embedding，它是token、position、segment三者embedding之和

从这里我们可以看到bert的embedding过程，同时还适配处理了`Conditional Layer Normalization`，[苏神博客](https://kexue.fm/archives/7124 )。

这里提一嘴，为什么三者相加呢？不怕信息混乱吗？

苏神在这里给的解释是：

    Embedding的数学本质，就是以one hot为输入的单层全连接。
    也就是说，世界上本没什么Embedding，有的只是one hot。 ”

所以你给三个拼接再送去下级网络，和加起来，并没有什么实质性区别。

### def apply_main_layers(self):

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L529 )

BERT的主体是基于Self-Attention的模块。顺序：Att --> Add --> LN --> FFN --> Add --> LN

这里是Bert的Transformer的最基本层（也就是Bert由12个这种层组成），由基类Transformer的[call](https://github.com/Sniper970119/bert4keras_document/tree/master/models#def-call )
进行循环调用

这里的LN依然适配了`Conditional Layer Normalization`，[苏神博客](https://kexue.fm/archives/7124 )。

* * *

## class ALBERT()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#776)

    class ALBERT(BERT)

ALBERT模型，继承Bert。

重新定义了`apply_main_layers`（核心层）和层名称映射（因为相比Bert，公共层参数了，所以映射也会发生变化。可以看到Albert的映射中并没有循环）。

### def apply_main_layers(self):

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L779 )

ALBERT的主体是基于Self-Attention的模块。顺序：Att --> Add --> LN --> FFN --> Add --> LN

其实这里除了命名（Bert的12层分别命名）之外，相比Bert没有什么变化。

由于Bert的[apply_embeddings](https://github.com/Sniper970119/bert4keras_document/tree/master/models#def-apply_embeddings )
已经处理了embedding和hidden size不符合的问题，因此Albert这里对嵌入压缩并不需要格外适配。

* * *

## class ALBERT_Unshared()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#893)

    class ALBERT_Unshared(BERT)

解开ALBERT共享约束，当成BERT用。

这个就可以只修改权重名映射了，因为“不共享”就和Bert一样了。embedding压缩Bert的基类也已经处理了。

* * *

## class NEZHA()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#933)

    class NEZHA(BERT)

华为推出的NAZHA模型。[论文链接](https://arxiv.org/abs/1909.00204 )

全称为“ **NE**ural contextuali**Z**ed representation for C**H**inese l**A**nguage understanding ”。

主要改进如下：

1.增加相对位置编码函数

- Bert中学习了绝对位置编码，Transformer中也是用了函数式编码。 NEZHA通过在注意力机制中引入相对位置的概念，提升了在NER等任务中的效果。

2.全词掩码

- 他减轻了预训练过程中掩码不分word pirce的弊端。 比如：playing在token部分会被分为play和##ing，而原生bert会随机的mask play或者##ing或者两者全部mask，而wwm则只会mask两者

3.混合精度训练

- 在训练过程中的每一个step，为模型的所有weight维护一个FP32的copy，称为Master Weights；在做前向和后向传播过程中，Master
  Weights会转换成FP16（半精度浮点数）格式，其中权重、激活函数和梯度都是用FP16进行表示，最后梯度会转换成FP32格式去更新Master Weights。
  由于float16的运算速度大于float32，因此能够显著提升训练速度。

4.优化器改进

- NEZHA使用了《Large Batch Optimization for Deep Learning：Training BERT in 76 minutes》
  （[def extend_with_layer_adaptation](https://github.com/Sniper970119/bert4keras_document/tree/master/optimizers#def-extend_with_layer_adaptation_v2 )
  ） 的一个优化器，它可以将预训练bert时间从三天降到76分钟。

从上面的改进就可以发现，模型端的改进主要就是位置编码。

因此NEZHA的embedding是token、segment两者embedding之和

### def apply_embeddings(self):

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L937 )

可以看到，并没有Position Embedding。同时依然适配了embedding压缩。

### def compute_position_bias(self):

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1077)

这里就是计算相对位置编码的地方。可以看到最后输出的维度为attention_key_size(bert base为64)。

调用[class RelativePositionEmbedding](https://github.com/Sniper970119/bert4keras_document/tree/master/layers#class-relativepositionembedding)

### def apply_main_layers(self):

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L997)

和其他的并没有什么太大差距，不同的是这里将position_bias（def compute_position_bias的返回）送入attention中。

这里送入Multi-Head-attention中的数据从3维上升到4维（或更高）。

详情查看[Multi-Head-attention](https://github.com/Sniper970119/bert4keras_document/tree/master/layers#class-MultiHeadAttention)


* * *

## class RoFormer()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#1096)

    class RoFormer(NEZHA)

旋转式位置编码的BERT模型。[苏神博客](https://kexue.fm/archives/8265 )

一个苏神（追一科技）自研的模型。

既然是“旋转式位置编码的BERT模型”，为什么继承NEZHA不继承BERT呢？

因为既然采用了“旋转式位置编码”，也就意味着同样是“相对位置编码”。

实际上，旋转式位置编码（Rotary Position Embedding，RoPE），是一种配合Attention机制能达到“绝对位置编码的方式实现相对位置编码”的设计。

因此，这种方式依然需要将绝对位置编码送入attention中。因此需要“借用”NEZHA中已经写好的位置编码（因为都没有进行position embedding）。

* * *

## class ELECTRA()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#1197)

    class ELECTRA(BERT)

Google推出的ELECTRA模型[论文](https://arxiv.org/abs/2003.10555 )

相比Bert，主要是将结构更改称为类强化学习的思路（但是不是），通过生成器和判别器来训练。[我的笔记](http://www.sniper97.cn/index.php/note/deep-learning/3842/ )

但是苏神这里的ELECTRA并不是完整模型，而只是判别器（只有在预训练过程中需要生成器）。

而原文的判别器也是bert base的模型，因此这里苏神只对模型特有的最后一层进行了一定的改变（`def apply_final_layers`）。

* * *

## class GPT()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1267)

    class GPT(LM_Mask, BERT)

GPT（[GPT-1](https://github.com/openai/finetune-transformer-lm))

可以看到，由于继承了`LM_Mask`，而`LM_Mask`复写了`compute_attention_bias`方法，更换为下三角矩阵，以达到Mask的效果。

### def apply_embedding()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1276)

    def apply_embeddings(self, inputs):

GPT的embedding是token、position、segment三者embedding之和。
跟BERT的主要区别是三者相加之后没有加LayerNormalization层。

* * *

## class GPT2()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1386)

    class GPT2(GPT)

构建GPT2模型，[GPT-2](https://github.com/openai/gpt-2)

### def get_inputs()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1390)

    def get_inputs(self):

GPT-2的输入。GPT2的输入是token_ids。

### def apply_embeddings()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1398)

    def apply_embeddings(self, inputs):

GPT2的embedding是token、position两者embedding之和。

### def apply_main_layers()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1433)

    def apply_main_layers(self, inputs, index):

GPT2的主体是基于Self-Attention的模块。

顺序：LN --> Att  --> Add --> LN --> FFN --> Add  

作为对比，这里贴出来Bert的：

顺序：Att --> Add --> LN --> FFN --> Add --> LN

### def apply_final_layers()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1508)

    def apply_final_layers(self, inputs):

GPT-2的剩余部分。

相比GPT，对了一个LN（和dropout），因此整体结构变为：

（LN --> Att  --> Add --> LN --> FFN --> Add ）*n --> LN --> Embedding

* * *

## class GPT2_ML()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1550)

    class GPT2_ML(GPT)



构建[GPT2_ML](https://github.com/imcaspar/gpt2-ml) 模型
GPT2_ML虽然号称GPT2，但是它的结构其实更接近GPT，它自称GPT2的原因大概是因为它开源的版本参数量达到了GPT2的15亿参数。

GPT2_ML的主体是基于Self-Attention的模块

顺序：Att  --> LN --> FFN --> Add --> LN

(GPT-2: LN --> Att  --> Add --> LN --> FFN --> Add 

Bert: Att --> Add --> LN --> FFN --> Add --> LN
)

* * *

## class T5_Base()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1740)

    class T5_Base(Transformer):


Google的T5模型（基类）

注意T5有两个版本，一开始放出来的版本称为t5.1.0，而后来放出了一个升级,版本称为t5.1.1。
两者结构略有不同，包括后来放出来的多国语言版T5也采用了t5.1.1的结构。

[t5.1.0](https://github.com/google-research/text-to-text-transfer-transformer)

[t5.1.1](https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md#t511)

[multilingual-t5](https://github.com/google-research/multilingual-t5)

* * *

## class T5_Encoder()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1861)

    class T5_Encoder(T5_Base):

Google的T5模型（Encoder）

### def apply_embeddings()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1874)

    def apply_embeddings(self, inputs):

T5的embedding只有token embedding，并把relative position embedding准备好，待attention使用。

### def apply_main_layers()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1906)

    def apply_main_layers(self, inputs, index):

T5的Encoder的主体是基于Self-Attention的模块

顺序：LN --> Att --> Add --> LN --> FFN --> Add

### def compute_position_bias()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L1906)

    def compute_position_bias(self, inputs=None):

T5相对位置编码。调用[def RelativePositionEmbeddingT5](Todo) 来计算相对位置编码。
* * *

## class T5_Decoder()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L2034)

    class T5_Decoder(LM_Mask, T5_Base):

Google的T5模型（Decoder）

### def apply_embeddings()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L2044)

    def apply_embeddings(self, inputs):

T5的embedding只有token embedding，并把relative position embedding准备好，待attention使用。

### def apply_main_layers()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L2091)

    def apply_main_layers(self, inputs, index):

T5的Decoder主体是基于Self-Attention、Cross-Attention的模块

顺序：LN --> Att1 --> Add --> LN --> Att2 --> Add -->  LN --> FFN --> Add

### def compute_position_bias()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L2293)

    def compute_position_bias(self, inputs=None):

T5相对位置编码。调用[def RelativePositionEmbeddingT5](Todo) 来计算相对位置编码。
* * *

## class T5()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L2322)

    class T5(T5_Base):

Google的T5模型（Encoder-Decoder）

* * *

分别调用[T5-Encoder](Todo) 和 [T5-Decoder](Todo)，构建一个完整的T5模型。

### def extend_with_language_model()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L2350)

    def extend_with_language_model(BaseModel):

* * *

### def extend_with_unified_language_model()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L2363)

    def extend_with_unified_language_model(BaseModel):

* * *

### def build_transformer_model()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/models.py#L2377)

    def build_transformer_model(
    config_path=None,
    checkpoint_path=None,
    model='bert',
    application='encoder',
    return_keras_model=True,
    **kwargs
    ):  

* * *
