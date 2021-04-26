# Sniper

### 写在前面

由于苏神开发的[bert4keras](https://github.com/bojone/bert4keras )并没有什么文档，苏神本人的[文档](https://bert4keras.spaces.ac.cn/ )写了个头就没动过了，取而代之的是很多例子。

这对框架的使用者造成了一定的困扰，事实上我在初识[bert4keras](https://github.com/bojone/bert4keras )时，因为没文档，一度选择了学习[transformers](https://github.com/huggingface/transformers )，当然[transformers](https://github.com/huggingface/transformers )是一个很好的框架，但是对于学习的人来说，并没有[bert4keras](https://github.com/bojone/bert4keras )来的实在，毕竟有苏神在带着。

本文主要为我在阅读和使用[bert4keras](https://github.com/bojone/bert4keras )源码时的思考，和使用方法一并写下（试图）作为一版文档吧。

顾名思义，其实本仓库并不能称作完整的文档，其实是一个教程，争取把用法和代码讲明白（我也能力有限，只能把我用过的和我学过的部分进行说明）。

已经做了[transformers](https://github.com/huggingface/transformers )的contributor努力做[bert4keras](https://github.com/bojone/bert4keras )的哈哈哈哈。

**本文可能会照搬很多苏神的文字和代码，如果苏神介意我就取消开源**

### 快速开始

下面是一个调用bert base模型来编码句子的简单例子： 

    from bert4keras.models import build_transformer_model
    from bert4keras.tokenizers import Tokenizer
    import numpy as np
    
    config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'
    
    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重
    
    # 编码测试
    token_ids, segment_ids = tokenizer.encode(u'语言模型')
    
    print('\n ===== predicting =====\n')
    print(model.predict([np.array([token_ids]), np.array([segment_ids])]))


