import argparse

from x_voice.model.utils import get_ipa_id, str_to_list_ipa_v6
from x_voice.train.datasets.ipa_v6_tokenizer import PhonemizeTextTokenizer


def main():
    parser = argparse.ArgumentParser(description="Convert input text to ipa_v6 token sequence.")
    parser.add_argument("--text", default="Hello, world!", type=str, help="Input text.")
    parser.add_argument("--language", default="en", type=str, help="Language code, e.g. en, zh, ja, ko, th.")
    args = parser.parse_args()

    ipa_language = get_ipa_id(args.language)
    tokenizer = PhonemizeTextTokenizer(language=ipa_language, with_stress=True)

    ipa_text = tokenizer(args.text)
    ipa_tokens = str_to_list_ipa_v6(ipa_text)

    print(f"language: {args.language}")
    print(f"ipa_language: {ipa_language}")
    print(f"input_text: {args.text}")
    print(f"ipa_text: {ipa_text}")
    print(f"ipa_tokens: {ipa_tokens}")


if __name__ == "__main__":
    main()
