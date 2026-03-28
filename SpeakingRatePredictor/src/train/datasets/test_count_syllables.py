import sys
from pathlib import Path

SPEAKING_RATE_ROOT = Path(__file__).resolve().parents[3]
MAVL_ROOT = SPEAKING_RATE_ROOT / "MAVL"

if str(MAVL_ROOT) not in sys.path:
    sys.path.insert(0, str(MAVL_ROOT))

import re
import unicodedata
import pyphen
from pythainlp.tokenize import syllable_tokenize
from process_syllable.japanese import split_syllables as ja_split_syllables

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

def count_syllables(text, lang="en"):
    def extract_pyphen_text(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        tokens = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", text, flags=re.UNICODE)
        return " ".join(tokens)

    def get_syllable_count(text: str, lang: str) -> int:
        if not text:
            return 0

        if lang in ["zh", "yue"]:
            chars = re.findall(r"[\u4e00-\u9fff]", text)
            return len(chars)
        
        if lang == "ko":
            chars = re.findall(r"[\uac00-\ud7a3]", text)
            return len(chars)

        if lang == "th":
            clean_text = "".join(
                ch for ch in text
                if unicodedata.category(ch)[0] in {"L", "M"} or ch.isspace()
            )
            return len(syllable_tokenize(clean_text))

        if lang == "ja":
            _, count = ja_split_syllables(text)
            return count

        if lang == "vi":
            tokens = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
            return len(tokens)

        pyphen_lang = PYPHEN_LANG_MAP.get(lang, "en_US")
        if pyphen_lang not in _PYPHEN_CACHE:
            try:
                _PYPHEN_CACHE[pyphen_lang] = pyphen.Pyphen(lang=pyphen_lang)
            except Exception:
                if "en_US" not in _PYPHEN_CACHE:
                    _PYPHEN_CACHE["en_US"] = pyphen.Pyphen(lang="en_US")
                pyphen_lang = "en_US"

        dic = _PYPHEN_CACHE[pyphen_lang]
        count = 0
        for word in text.split():
            if word:
                count += len(dic.inserted(word).split("-"))
        return count

    if lang in {"th", "ja"}:
        return get_syllable_count(text, lang)

    if lang in {"zh", "ko", "yue", "vi"}:
        return get_syllable_count(text, lang)

    clean_text = extract_pyphen_text(text)
    return get_syllable_count(clean_text, lang)

if __name__ == "__main__":
    # 测试用例：30 种语言
    test_cases = [
        ('bg', 'Здравей свят'),                    # 保加利亚语: 你好世界
        ('cs', 'Ahoj světe'),                      # 捷克语
        ('da', 'Hej verden'),                      # 丹麦语
        ('de', 'Hallo Welt'),                      # 德语
        ('el', 'Γειά σου κόσμε'),                  # 希腊语
        ('en', 'Hello world'),                     # 英语
        ('es', '¡Hola mundo!'),                    # 西班牙语
        ('et', 'Tere maailm'),                     # 爱沙尼亚语
        ('fi', 'Hei maailma'),                     # 芬兰语
        ('fr', 'Bonjour le monde'),                # 法语
        ('hr', 'Pozdrav svijete'),                 # 克罗地亚语
        ('hu', 'Helló világ'),                     # 匈牙利语
        ('id', 'Halo dunia'),                      # 印尼语
        ('it', 'Ciao mondo'),                      # 意大利语
        ('ko', '안녕하세요 세계'),                   # 韩语
        ('lt', 'Sveikas pasauli'),                 # 立陶宛语
        ('lv', 'Sveika pasaule'),                  # 拉脱维亚语
        ('mt', 'Hello dinja'),                     # 马耳他语
        ('nl', 'Hallo wereld'),                    # 荷兰语
        ('pl', 'Cześć świecie'),                   # 波兰语
        ('pt', 'Olá mundo'),                       # 葡萄牙语
        ('ro', 'Salut lume'),                      # 罗马尼亚语
        ('ru', 'Привет мир'),                      # 俄语
        ('sk', 'Ahoj svet'),                       # 斯洛伐克语
        ('sl', 'Pozdravljen svet'),                # 斯洛文尼亚语
        ('sv', 'Hej världen'),                     # 瑞典语
        ('th', 'สวัสดีโลก。'),                        # 泰语
        ('vi', 'Xin chào thế giới'),               # 越南语
        ('zh', '你好世界123'),                      # 中文
        ('ja', 'こんにちは世界'),                    # 日语
    ]

    # 运行测试
    print(f"{'语言':<6} {'文本':<25} {'音节数':<6}")
    print("-" * 45)

    for lang, text in test_cases:
        syllables = count_syllables(text, lang=lang)
        print(f"{lang:<6} {text:<25} {syllables:<6}")
