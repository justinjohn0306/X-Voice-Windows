# SRP Preprocess补充说明 (for Multilingual F5-TTS)
## 数据集大小
- 训练集：从多语言数据集里每个语言抽250h的数据出来，用作计算SRP
- 验证集：从每个语言再抽100条sample出来，做验证集

## filter
### 1. 字符串验证函数
下面是`字符串验证函数`, 核心作用是确保保留的训练样本中对应的transcript必须只有 **正常的字符**和 **符合要求的标点**

```py
def check_valid_chars(input_str):
    valid_punctuation = '\'",.?!;:。，、！？；：「」『』【】'
    
    for c in input_str:
        # Check if it's a letter (any language)
        if c.isalpha():
            continue
        # Check if it's a specified punctuation mark
        if c in valid_punctuation:
            continue
        # Check if it's a space or whitespace
        if c.isspace():
            continue
        return False
        
    return True
```

## syllables计算
1. 先用`pyphen.Pyphen`计算过滤后符合要求sample的syllable数（特殊符号 和 标点符号不包括在音节计算里）
2. 然后用map_to_class转换成离散类别
   ```py
   def map_to_class(speed, delta=0.25):
       return round(speed / delta) * delta
   ```

## 输出的文件
1. "raw.arrow" 给训练集
2. "raw_val.arrow" 给验证集
3. "duration.json" 给训练集
4. "duration_val.json" 给验证集
5. 还需要额外输出speed_syllables类别的统计信息 和 直方图到data

arrow的四个字段：
- audio_path：音频绝对路径
- text：文本内容
- duration：时长ground truth
- speed_syllables：计算并映射后的离散类别
- lang: 语言类别

---

## 语言
1	bg保加利亚
2	cs捷克
3	da丹麦
4	de德国
5	el希腊
6	en
7	es西班牙
8	et爱沙尼亚
9	fi芬兰
10	fr法国
11	hr克罗地亚
12	hu匈牙利
13	id印度
14	it意大利
15	ko韩国
16	lt立陶宛
17	lv拉脱维亚
18	mt马耳他
19	nl荷兰
20	pl波兰
21	pt葡萄牙
22	ro罗马尼亚
23	sk斯洛伐克
24	sl斯洛文尼亚
25	sv瑞典
26	th泰国
27	vi越南
28	zh
29	ja日本
30	ru俄罗斯

---

## 泰语和日语
泰语用pythainlp
```py
from pythainlp.tokenize import syllable_tokenize
return len(syllable_tokenize(text))
```

日语按https://github.com/k1064190/MAVL/blob/main/process_syllable/japanese.py这个来改
>[japanese.py](/Users/qingyu/Desktop/TTS/code/Multilingual_F5-TTS/SpeakingRatePredictor/MAVL/process_syllable/japanese.py)

---

## 训练链路
1. dataset和collate_fn额外接收"lang"
   - `src/model/dataset.py`里的`CustomDataset.__getitem__()`需要从arrow里读取`lang`并返回
   - `src/model/dataset.py`里的`collate_fn_sp()`需要把batch中的`lang`一起打包返回，比如`langs`
   - `src/model/dataset.py`里validation用到的`count(text)`还是旧的英中逻辑，需要替换成和`prepare_multiilingual_speed.py`一致的多语言`count_syllables(text, lang)`
2. model/trainer.py#L397 validate 逻辑需要修改，先分开计算30种语言的val得到一个accuracy，再得到一个总的accuracy，依然记录到output_file = f"{log_validation_path}/val_{global_update}.jsonl"；并且`summary_file = f"{log_validation_path}/accuracy_summary.jsonl"`通过比较总的accuracy得到最好的step
   - `validate()`里要从batch中取出`lang`
   - validation结果需要按`lang`分别统计`correct`、`total`和`accuracy`
   - 在30种语言accuracy之外，再计算一个总的accuracy
   - `val_{global_update}.jsonl`里既要保留逐条sample结果，也要写入per-language summary和overall summary
   - `accuracy_summary.jsonl`仍然用overall accuracy选best step，但建议同时记录每个step的per-language accuracy，方便分析
   - 如果当前模型只是在validation阶段按语言分组统计，那么模型本身可以先不接收`lang`；如果后续训练也要显式使用语言信息，再继续改`SpeedPredictor`
