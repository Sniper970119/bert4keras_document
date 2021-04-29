# Sniper

`snippets`为一些小工具，该文件内置了数据处理时可能使用到的工具。

### def strQ2B()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L27)

    def strQ2B(ustring)

其中：

|参数| 说明|
|:-----  |-----|
|ustring|全角字符串:str|

返回：

|参数| 说明|
|:-----  |-----|
|rstring  |半角字符串:str|

这个方法负责将全角字符串转换为半角字符串。

### def string_matching()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L49)

    string_matching(s, keywords)

其中：

|参数| 说明|
|:-----  |-----|
|s|字符串:str|
|keywords|关键字:list|

返回：

|参数| 说明|
|:-----  |-----|
|flag  |是否包含关键字:bool|

判断s中是否包含关键字。


### def convert_to_unicode()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L58)

    def convert_to_unicode(text, encoding='utf-8', errors='ignore')
    
其中：

|参数| 说明|
|:-----  |-----|
|text|字符串:str|
|encoding|输入字符串编码方式:str|
|errors|忽略警告:str|

返回：

|参数| 说明|
|:-----  |-----|
|text  |解码后的文本:str|

字符串unicode解码。

### def convert_to_str()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L70)

    def convert_to_str(text, encoding='utf-8', errors='ignore')
    
其中：

|参数| 说明|
|:-----  |-----|
|text|字符串:str|
|encoding|输出字符串编码方式:str|
|errors|忽略警告:str|

返回：

|参数| 说明|
|:-----  |-----|
|text  |编码后的文本:str|

字符串编码。

### class open()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L82)

这个方法主要是兼容py2和py3，引入索引，方便大文件读取。

但是我并没有看见过苏神使用过这个方法（使用索引），本人代码复现的时候也是直接不用这个open。

因此暂无。

### def parallel_apply()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L189)

    def parallel_apply(
        func,
        iterable,
        workers,
        max_queue_size,
        callback=None,
        dummy=False,
        random_seeds=True
    )
    
其中：

|参数| 说明|
|:-----  |-----|
|func|单个数据处理的方法:function|
|iterable|数据:list|
|workers|线程 or 进程 数:int|
|max_queue_size|最大队列长度:int|
|callback|回调函数:function|
|dummy|是否为多进程:bool|
|random_seeds|随机种子:bool|

返回：

|参数| 说明|
|:-----  |-----|
|res  |多线程处理后的结果:list|

多线程（进程）处理框架，可以用于多线程（进程）数据处理。

windows系统中仅支持多线程；Linux系统中支持多线程和多进程。

`func`为每条的数据处理方法。

`iterable`为一个可迭代结果，事实上，它不仅可以是list，任何迭代结构（可以用于enumerate中的）都可以，列表、元组、字符串等，当然，tqdm也可以。

`workers`线程（进程）数。

`max_queue_size`最大队列长度，每个线程（进程）队列长度。

`callback`回调函数，每个数据处理完成后的回调函数

`dummy`为True为多进程，False为多线程（Windows仅支持False，原因是python在windows中使用多进程必须在main结构下）。

`random_seeds`每个线程的随机种子。

example：

    # input:
    from tqdm import tqdm
    from bert4keras.snippets import parallel_apply
    
    def fun(a):
        print(a)
        return a
    
    def fun1(a):
        print('callback', a)
    
    # call with return
    res = parallel_apply(
        func=fun,
        iterable=tqdm(range(5), desc=u'转换数据'),
        workers=2,
        max_queue_size=2,
        dummy=True,
    )
    print(res)

    # output:
    转换数据: 100%|██████████| 5/5 [00:00<00:00, 208.29it/s]
    0
    1
    2
    3
    4
    [0, 1, 2, 3, 4]

    # call with callback
    res = parallel_apply(
        func=fun,
        iterable=tqdm(range(5), desc=u'转换数据'),
        workers=2,
        max_queue_size=2,
        dummy=True,
        callback=fun1
    )
    print(res)
    
    # output:
    转换数据: 100%|██████████| 5/5 [00:00<00:00, 185.14it/s]
    0
    1
    2
    3
    callback 0
    4
    callback 1
    callback 2
    callback 3
    callback 4
    None
    

### def sequence_padding()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L270)

    def sequence_padding(inputs, length=None, padding=0, mode='post')

其中：

|参数| 说明|
|:-----  |-----|
|inputs|输入序列:list|
|length|填充长度:int|
|padding|填充值:int|
|mode|填充模式:str|


返回：

|参数| 说明|
|:-----  |-----|
|res  |填充后的结果:ndarry|

将序列padding到同一长度

`inputs`输入数据。

`length`将所有的列表填充到该长度，为None则取整个列表中最大长度 `max([len(x) for x in inputs])`。

`padding`填充值，填充部分的值，默认0。

`mode`填充模式，post 和pre两种，分别为向后填充和向前填充，默认为post。

返回填充后的ndarry。

实际上这个方法苏神还当作np.array()用了，对于不需要填充只需要转化为ndarry的变量他也调用的`sequence_padding`。
也确实有这个功能，毕竟最后返回的是ndarry。


### def truncate_sequences()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L292)

    def truncate_sequences(maxlen, index, *sequences)
    
其中：

|参数| 说明|
|:-----  |-----|
|maxlen|输入序列:list|
|index|填充长度:int|
|*sequences|填充值:int|


返回：

|参数| 说明|
|:-----  |-----|
|sequences  |填充后的结果:ndarry|

这个函数是用来做序列截断的。

`maxlen`为截断的最大长度（所有句子长度和）。

`index` index为删除的位置（对每个句子，删除同一个位置，从而做到句子长度缩小）

`*sequences`多参，句子对。

这个函数主要是在[encode](https://github.com/Sniper970119/bert4keras_document/tree/master/tokenizers#def-encode )中使用的。


### def text_segmentate()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L305)

    def text_segmentate(text, maxlen, seps='\n', strips=None)
    
其中：

|参数| 说明|
|:-----  |-----|
|text|输入序列:str|
|maxlen|填充长度:int|
|seps|填充值:str|
|strips|填充值:str|


返回：

|参数| 说明|
|:-----  |-----|
|text  |切割后的结果:list|

递归拆分文本，相比通过逗号和句号分词提供更多的拆分可能。

`text`为需要被拆分的文本。

`maxlen`为拆分的最大长度，当文本长度不足`maxlen`时不做拆分。

`seps`拆分字符，为字符串类型，如`？！`则代表分别依据？和！进行拆分。比如“你好？你好！”则会被拆成两个你好。
**其实这里感觉替换成列表更好，并不需要更改逻辑代码，
还能适配多字符的拆分（现在只支持单字符，比如想根据'...'进行拆分是不行的，只能对'.'进行拆分）**

`strips`需要被去掉的字符，为字符串类型，拆分后的结果如果首位有这些字符则会被去掉。

而该方法的返回值可能是具有多个长度的列表（因为可能进行了拆分，也可能不进行拆分）

example：

    text = '贝贝好爱干净！每天出门都要洗澡。还喜欢喝蒙牛！不喜欢蹲地方~喜欢坐凳子上还喜欢和我坐在一起~'
    texts = text_segmentate(text, 1, u'\n。；：，！~', u'。')
    # output:
    ['贝贝好爱干净！', 
    '每天出门都要洗澡', 
    '还喜欢喝蒙牛！', 
    '不喜欢蹲地方~', 
    '喜欢坐凳子上还喜欢和我坐在一起~']
    

### def is_one_of()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L327)

    def is_one_of(x, ys)
    
其中：

|参数| 说明|
|:-----  |-----|
|x|keyword:any|
|ys|列表:list|



返回：

|参数| 说明|
|:-----  |-----|
|res  |x是否在ys中:bool|

判断x是否在ys之中，等价于x in ys，但有些情况下x in ys会报错


## class DataGenerator(object)

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L337)

    def __init__(self, data, batch_size=32, buffer_size=None)
    
其中：

|参数| 说明|
|:-----  |-----|
|data|数据:list|
|batch_size|批大小:int|
|buffer_size|缓存大小:int|


`data`为所有的数据，一般为从文件读取后的数据列表。

`batch_size`数据批大小。

`buffer_size`缓存大小，默认为批大小*1000。

该类为数据生成器，用来将数据分批。通常在该类中将数据tokenizer然后分批次返回。

### def sample()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L354)

    def sample(self, random=False)  
    
其中：

|参数| 说明|
|:-----  |-----|
|random|是否随机:bool|


返回：

|参数| 说明|
|:-----  |-----|
|flag  |是否为最后一个数据:bool|
|data  |数据:bool|

这个方法就是（随机）取出一条数据，同时返回这条数据是否为最后一条。

这里虽然默认random为False，但是有意思的是：其他地方(def forfit())调用该方法时，默认为True
（也就是这里的random=False实际上是random=True），不知道是苏神有意为之还是一个小彩蛋。

当然啦，你直接调用这个方法当然还是False。也不是bug，只是好奇，在这里提一嘴。

这个就是（随机）遍历数据，如果到了batch size大小，就yield等待返回。

### def \_\_iter__()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L390)

    def __iter__(self, random=False)

其中：

|参数| 说明|
|:-----  |-----|
|random|是否随机:bool|

这个方法是需要子类实现的，用来定义每一批次数据的具体实现。

通常为 读取数据->tokenize->封装batch->返回数据->读取数据

example:

    from bert4keras.snippets import sequence_padding, DataGenerator
    class data_generator(DataGenerator):
        def __iter__(self, random=False):
            batch_token_ids, batch_segment_ids, batch_labels = [], [], []
            # 读取数据
            for is_end, (source, target, label) in self.sample(random):
                # tokenize
                token_ids, segment_ids = tokenizer.encode(
                    source, target, maxlen=maxlen
                )
                # 封装batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
                # 返回数据
                if len(batch_token_ids) == self.batch_size or is_end:
                    # 序列填充（这里填充长度为每一批最大长度，因为没指定maxlen）
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    # yield返回
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                    batch_token_ids, batch_segment_ids, batch_labels = [], [], []


### def forfit()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L393)

    def forfit(self, random=True)
    
其中：

|参数| 说明|
|:-----  |-----|
|random|是否随机:bool|

这个方法就是用来返回数据的，model.fit中，可以传入一个迭代器，巧了，这个就是。

我们可以看下这个的源码，反正也不长，这个random默认为True。

        def forfit(self, random=True):
            while True:
                for d in self.__iter__(random):
                    yield d

我们可以看到，实际上从 forfit -> \_\_iter__ -> sample ，random为True，但是\_\_iter__ 和 sample 中random默认
都是False。咱也不晓得为啥。

## class ViterbiDecoder(object)

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L438)

我没用过，也没看苏神用过。学习到之后再来补充。

## class AutoRegressiveDecoder(object)

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L484)

    def __init__(self, start_id, end_id, maxlen, minlen=1)
    
其中：

|参数| 说明|
|:-----  |-----|
|start_id|开始id:int|
|end_id|结束id:int|
|maxlen|最大长度:int|
|minlen|最小长度:int|

该类为一个自回归生成模型解码的基类。

`start_id` 为解码的起始id，一般为CLS。

`end_id` 为解码的结束id，一般为SEP。

`maxlen` 文本最大长度。

`minlen` 文本最小长度。

### def last_token()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L539)

    def last_token(self, model)
    

其中：

|参数| 说明|
|:-----  |-----|
|model|模型:tf.keras.model.Model|

创建一个只返回最后一个token输出的新Model。

### def wraps()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L500)

    @staticmethod
    def wraps(default_rtype='probas', use_states=False)

其中：

|参数| 说明|
|:-----  |-----|
|default_rtype|模型:tf.keras.model.Model 或 未知|
|use_states|模型:tf.keras.model.Model 或 未知|


返回：

|参数| 说明|
|:-----  |-----|
|model|模型:tf.keras.model.Model|

一个静态方法，用来完善predict函数。

`default_rtype`为`probas`时为随机采样调用，`logits`为`beam_search`时调用；

该方法实际上主要就是model.predict()然后完善了一下输出。

### def last_token()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L539)

    def last_token(self, model)

其中：

|参数| 说明|
|:-----  |-----|
|model|模型:tf.keras.model.Model 或 未知|


返回：

|参数| 说明|
|:-----  |-----|
|model|模型:tf.keras.model.Model|

创建一个只返回最后一个token输出的新Model。

emm这个方法也挺奇妙的，感觉不到太大用处，个人认为可能是用于非keras模型时，这里返回一个keras.Model的模型。

### def beam_search()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L560)

    def beam_search(self, inputs, topk, states=None, temperature=1, min_ends=1)
    
其中：

|参数| 说明|
|:-----  |-----|
|inputs|输入:list|
|topk|beam size:int|
|states|未知:None|
|temperature|未知：float|
|min_ends|出现结束标志的次数（猜的） : int|


返回：

|参数| 说明|
|:-----  |-----|
|res|n个解码序列组成的列表:list|


beam search解码。

`inputs`输入的序列，如果没有输入则为空列表
`topk`topk即beam size
`states`状态，我现在只见过None，我也不知道这玩意具体是干嘛的。
`temperature` 默认为1，是[predict](https://github.com/Sniper970119/bert4keras_document/tree/master/snippets#def-wraps )中的一个参数，用来控制结果的softmax比例。
`min_ends`从代码阅读结果来看，应该是最小的结束标记次数，默认为1（比如生成nsp那种句子，则为2）。

### def random_sample()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L598)

        def random_sample(
        self,
        inputs,
        n,
        topk=None,
        topp=None,
        states=None,
        temperature=1,
        min_ends=1
    )
    
其中：

|参数| 说明|
|:-----  |-----|
|inputs|输入:list|
|n|输出个数:int|
|topk|beam size:int|
|topp|随机概率:int|
|states|未知:None|
|temperature|未知：float|
|min_ends|出现结束标志的次数（猜的）: int|


返回：

|参数| 说明|
|:-----  |-----|
|res|n个解码序列组成的列表:list|

随机采样解码。

`inputs`输入的序列，如果没有输入则为空列表
`n` 输出个数
`topk`非None的topk表示每一步只从概率最高的topk个中采样 
`topp`非None的topp表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样
`states`状态，我现在只见过None，我也不知道这玩意具体是干嘛的。
`temperature`默认为1，是[predict](https://github.com/Sniper970119/bert4keras_document/tree/master/snippets#def-wraps )中的一个参数，用来控制结果的softmax比例。
`inputs`非None的topp表示每一步只从概率最高的且概率之和刚好达到topp的若干个token中采样


### def longest_common_substring()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L705)

    def longest_common_substring(source, target)
    
其中：

|参数| 说明|
|:-----  |-----|
|source|文本1：str|
|target|文本2:str|


返回：

|参数| 说明|
|:-----  |-----|
|l  |最大公共子串长度:int|
|子串位置  |数据:list|

查找最长公共子串。

`source`这里有一点出入吧，其实不应该命名为source和target，就是s1和s2的关系，找子串嘛。

`target`：s2。

最终返回一个最长子串长度和一个具有四个元素的列表，分别代表子串在s1和s2的start和end。

这个算法可能还算是个暴力算法，并没有用例如KMP这类优化后的子串算法。


### def longest_common_subsequence()

[&SOURCE](https://github.com/bojone/bert4keras/blob/master/bert4keras/snippets.py#L721)

    def longest_common_subsequence(source, target)
    
其中：

|参数| 说明|
|:-----  |-----|
|source|文本1：str|
|target|文本2:str|


返回：

|参数| 说明|
|:-----  |-----|
|l  |最大公共子串长度:int|
|映射关系（映射对组成的列表）  |数据:list|

最长公共子序列

这个算法如果没有bug的话，只能说这个算法比较鸡肋（或者我还是没想到这个的应用场景）。

我刚开始想的是最长公共子串的序列版本，结果并不是。

比如：

    s1 = [1, 2, 3, 4, 5]
    s2 = [3, 1, 2, 4, 1]
    longest_common_subsequence(s1, s2)
    
    # output：
    [(0, 0), (1, 2), (3, 3)]

我以为会输出\[1,2]这个公共子串，结果只是输出了一样的位置(s1: idx0  == s2: idx0; s1: idx1 == s2: idx2)

那问题来了，s1: idx0  == s2: idx5 为啥没有呢？

不是很想找这个问题，等遇到苏神使用的时候再具体学习吧。