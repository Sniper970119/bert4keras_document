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
|func|字符串:function|
|iterable|输出字符串编码方式:list|
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
    texts = text_segmentate(text, 1, u'\n。；：，！~',u'。')
    # output:
    ['贝贝好爱干净！', 
    '每天出门都要洗澡', 
    '还喜欢喝蒙牛！', 
    '不喜欢蹲地方~', 
    '喜欢坐凳子上还喜欢和我坐在一起~']