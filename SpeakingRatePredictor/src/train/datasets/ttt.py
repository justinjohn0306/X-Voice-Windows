import re
import unicodedata
import pyphen

PYPHEN_LANG_MAP = {
    "bg": "bg_BG", # 保加利亚语
    "cs": "cs_CZ", # 捷克语
    "da": "da_DK", # 丹麦语
    "de": "de_DE", # 德语
    "el": "el_GR", # 希腊语
    "en": "en_US", # 英语
    "es": "es_ES", # 西班牙语
    "et": "et_EE", # 爱沙尼亚语
    "fi": "fi_FI", # 芬兰语
    "fr": "fr_FR", # 法语
    "hr": "hr_HR", # 克罗地亚语
    "hu": "hu_HU", # 匈牙利语
    "id": "id_ID", # 印尼语
    "it": "it_IT", # 意大利语
    "lt": "lt_LT", # 立陶宛语
    "lv": "lv_LV", # 拉脱维亚语
    "mt": "mt_MT", # 马耳他语 (如果没有字典会自动回退)
    "nl": "nl_NL", # 荷兰语
    "pl": "pl_PL", # 波兰语
    "pt": "pt_PT", # 葡萄牙语
    "ro": "ro_RO", # 罗马尼亚语
    "ru": "ru_RU", # 俄语
    "sk": "sk_SK", # 斯洛伐克语
    "sl": "sl_SI", # 斯洛文尼亚语
    "sv": "sv_SE", # 瑞典语
}

_PYPHEN_CACHE = {}

def count_syllables(text, lang='en'):
    def get_syllable_count(text: str, lang: str) -> int:
        """
        计算文本音节数。
        - 中文/日文/韩文 (CJK): 按字符数计算
        - 其他语言: 使用 pyphen 连字符算法计算
        """
        if not text:
            return 0
            
        if lang in ["zh", "ja", "ko", "th", "yue"]:
            # 简单清洗一下，只算有效字符
            clean_text = re.sub(r"\s+", "", text) 
            # print(clean_text, len(clean_text))
            return len(clean_text)
        
        if lang == "vi":
            return len(text.split())
        # 获取 Pyphen 对应的字典代码
        pyphen_lang = PYPHEN_LANG_MAP.get(lang, "en_US") # 默认回退到英语规则
        if pyphen_lang not in _PYPHEN_CACHE:
            try:
                _PYPHEN_CACHE[pyphen_lang] = pyphen.Pyphen(lang=pyphen_lang)
            except Exception:
                # 如果加载失败（比如不支持的语言），回退到英语通用规则
                if "en_US" not in _PYPHEN_CACHE:
                    _PYPHEN_CACHE["en_US"] = pyphen.Pyphen(lang="en_US")
                pyphen_lang = "en_US"
        dic = _PYPHEN_CACHE[pyphen_lang]
        # Pyphen 的 inserted 方法会在音节间插入连字符，例如 "hello" -> "hel-lo"
        # 我们分割连字符并计算数量；先分词，再对每个词算音节
        count = 0
        words = text.split()
        for word in words:
            if not word.strip(): continue
            # inserted 返回 "hy-phen-a-tion"
            syllables = dic.inserted(word).split('-')
            count += len(syllables) 
        return count
    def extract_word_tokens(text):
        # 只保留"字母序列"，自动过滤数字、标点、特殊符号
        # 返回用空格连接的字符串
        tokens = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
        return " ".join(tokens)
    
    clean_text = extract_word_tokens(text)
    return get_syllable_count(clean_text, lang)