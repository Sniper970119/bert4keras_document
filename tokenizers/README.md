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

## class Tokenizer(TokenizerBase)

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

用于类Bert模型的tokenizer。

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
    
### def encode()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/tokenizers.py#L113)

        def encode(
        self,
        first_text,
        second_text=None,
        maxlen=None,
        pattern='S*E*E',
        truncate_from='right'
    )
    
该方法为`Tokenizer`的父类`TokenizerBase`中的方法。

其中：

|参数| 说明|
|:-----  |-----|
|first_text|第一个文本:str|
|second_text|第二个文本:str|
|maxlen|最大长度:int|
|pattern|规则:str|
|truncate_from|截断方向 :str|

返回：

|参数| 说明|
|:-----  |-----|
|token_ids  |token ids:list|
|segment_ids  |segment_ids:list|

对文本进行编码（变成token）。

`first_text` 第一个文本（对应bert中nsp任务的两个句子，虽然并不一定是nsp任务）。

`second_text`第二个文本。

`maxlen`返回数据的最大长度，超过截断（比如bert base的512）。

`pattern`对于first text 和second text的拼接规则,S代表CLS，E代表SEP，代表两句话的拼接规则，比如默认的S\*E\*E则为CLS sentence 1 SEP sentence 2 SEP，而S\*ES\*E 则为 CLS sentence 1 SEP CLS sentence 2 SEP。

`truncate_from`截断方向，对于超过最大长度的数据，截头部（left）还是截尾部（right）当然还支持通过一直删某个中间位置来截取（某一int值），
截取方法详见[truncate_sequences](https://github.com/Sniper970119/bert4keras_document/tree/master/snippets#def-truncate_sequences )。

**注意，由于bert4keras只提供了两局的分割，既[CLS]sentence1[SEP]sentence2[SEP]，如果用户有多句输入的需求，比如[CLS]sentence1[SEP]sentence2[SEP]sentence3[SEP],原生bert4keras无法做到[因为encode的实现逻辑将“[SEP]"视为了多个字符]，需要复写encode[支持更个性化的tokenize方式]，或者只需要简单的复写tokenize即可**

example：

    class CustomTokenizer(Tokenizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def tokenize(self, text, maxlen=None):
            """分词函数
            """
            tokens = [
                self._token_translate.get(token) or token
                for token in self._tokenize(text)
                ]
            for i in range(len(tokens)):
                # 将#转换为SEP，只有这里的SEP会被识别为分隔符
                if tokens[i] == '#':
                    tokens[i] = '[SEP]'
            if self._token_start is not None:
                tokens.insert(0, self._token_start)
            if self._token_end is not None:
                tokens.append(self._token_end)

            if maxlen is not None:
                index = int(self._token_end is not None) + 1
                truncate_sequences(maxlen, -index, tokens)

            return tokens

上述代码就可以通过输入tokenizer sentence1#sentence2#sentence3 来实现[CLS]sentence1[SEP]sentence2[SEP]sentence3[SEP]

### def decode()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/tokenizers.py#L215)

    def decode(self, ids, tokens=None)
    
解码器，将token变为可读的文本。

其中：

|参数| 说明|
|:-----  |-----|
|ids|token ids:list|
|tokens|映射后的文本:list|

返回：

|参数| 说明|
|:-----  |-----|
|text  |可读文本:str|

`ids`:encode之后（或模型输出）的token ids列表，例如`[101，770，102]`，返回‘你’。

`tokens`：这个比较简单，当tokens不为None时，ids无效。方法变为将tokens的字符拼接、处理，作为可读文本返回。
例如`['[CLS]'，'你'，'[SEP]']` 则会返回‘你’。

## class SpTokenizer(TokenizerBase)

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/tokenizers.py#L404)

用于类T5模型的tokenizer。

### def encode()

详见Tokenizer的[encode](https://github.com/Sniper970119/bert4keras_document/tree/master/tokneizers#def-encode )

这两个共同继承自TokenizerBase，并在TokenizerBase中完成实现。

这个之所以单独写出来主要是因为T5的tokenizer使用了sentencepiece，与bert有区别。这个sentencepiece我现在还没有研究，回头研究了回来补。
