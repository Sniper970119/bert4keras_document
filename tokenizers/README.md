# Sniper

`tokenizers` 为tokenizer相关工具，该文件内置了所有在生成token时可能使用到的工具。

### def load_vocab()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/tokenizers.py#L11)

    load_vocab(dict_path, encoding='utf-8', simplified=False, startswith=None) 
    
其中：

|参数| 说明|
|:-----  |-----|
|dict_path|词汇表路径:str|
|encoding|编码方式:str|
|simplified|是否简化 :bool|
|startswith| 开始标记:list|

返回：

|参数| 说明|
|:-----  |-----|
|token_dict|格式为 {keyword:idx,keyword:idx} 的字典:dict|
|keep_tokens（optional）|列表，精简后的词汇表在源词汇表中的映射:list|

这个方法就是读取bert中的`vovab.txt`文件，返回一个字典，格式为`{keyword:idx,keyword:idx}`。

`dict_path` 为`vovab.txt`文件的路径。

`encoding` 为`vovab.txt`的编码方式。

`simplified`为True则开启精简词汇表模型，通过去除CJK类字符和标点符号，将之前的21128（bert-chinese）个词精简到13584个，
从而将词汇表变小（随之而来的就是embedding变小，通过将21128\*768切片成13584\*768）,同时，返回`token_dict`和`keep_tokens`,
`token_dict`负责生成token，`keep_tokens`则传入模型构建中，来对embedding进行切片。

`startswith`则为一个列表，在`simplified`为True时使用，用来保留一些特殊字符（比如[PAD], [UNK], [CLS], [SEP]）。

example：

    token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
   
### def save_vocab()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/tokenizers.py#L48)

    save_vocab(dict_path, token_dict, encoding='utf-8')
    
其中：

|参数| 说明|
|:-----  |-----|
|dict_path|词汇表保存路径:str|
|token_dict|需要被保存的词汇表:dict |
|encoding|编码方式:str|


保存词汇表（比如精简后的）。

### class Tokenizer()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/tokenizers.py#L181)

        def __init__(
        self,
        token_dict, 
        do_lower_case=False, 
        word_maxlen=200
        token_start='[CLS]',
        token_end='[SEP]',
        pre_tokenize=None,
        token_translate=None,
    )

其中：

|参数| 说明|
|:-----  |-----|
|token_dict|vocab.txt的路径:str|
|do_lower_case|是否转化为小写:bool|
|word_maxlen|单词最大长度:int|
|token_start|token开始标记（CLS）:str|
|token_end|token结束标记（SEP） :str|
|pre_tokenize|预分词，在预分词的基础上进行token:function|
|token_translate|token转换:dict|

`token_dict`为vacab.txt的路径，方法内调用[load_vocab()](https://github.com/Sniper970119/bert4keras_document/tree/master/tokenizers#def-load_vocab )获得词汇表。

`do_lower_case`是否全部转化为小写（bert case和uncase）。

`word_maxlen`单词最大长度，由于使用了细粒度拆次因此，词中词，比如北京大学中有北京 和 大学，定义最大长度，比如word_maxlen=3，则不对北京大学进行拆分。

`token_start`token开始标记（默认CLS）。

`token_end`token结束标记（默认SEP）。

`pre_tokenize`为一个预分词方法，可以是一个jieba分词等方法，用来先将句子分词然后做token。

`token_translate`为token替换字典，比如需要将所有的CLS替换为SEP({101:102})。

example:

    tokenizer = Tokenizer(token_dict, do_lower_case=True)

    tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
    )
    
### class SpTokenizer()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/tokenizers.py#L404)

暂无