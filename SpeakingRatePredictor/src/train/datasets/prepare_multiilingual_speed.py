import argparse
from collections import Counter
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

import matplotlib
import pyphen
from datasets.arrow_writer import ArrowWriter
from pythainlp.tokenize import syllable_tokenize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPEAKING_RATE_ROOT = Path(__file__).resolve().parents[3]
MAVL_ROOT = SPEAKING_RATE_ROOT / "MAVL"

if str(MAVL_ROOT) not in sys.path:
    sys.path.insert(0, str(MAVL_ROOT))

from process_syllable.japanese import split_syllables as ja_split_syllables

TRAIN_HOURS_PER_LANG = 250
VAL_SAMPLES_PER_LANG = 100
VALID_PUNCTUATION = '\'",.?!;:。，、！？；：「」『』【】'

PYPHEN_LANG_MAP = {
    "bg": "bg_BG",
    "cs": "cs_CZ",
    "da": "da_DK",
    "de": "de_DE",
    "el": "el_GR",
    "en": "en_US",
    "es": "es_ES",
    "et": "et_EE",
    "fi": "fi_FI",
    "fr": "fr_FR",
    "hr": "hr_HR",
    "hu": "hu_HU",
    "id": "id_ID",
    "it": "it_IT",
    "lt": "lt_LT",
    "lv": "lv_LV",
    "mt": "mt_MT",
    "nl": "nl_NL",
    "pl": "pl_PL",
    "pt": "pt_PT",
    "ro": "ro_RO",
    "ru": "ru_RU",
    "sk": "sk_SK",
    "sl": "sl_SI",
    "sv": "sv_SE",
}

_PYPHEN_CACHE = {}


def check_valid_chars(input_str: str) -> bool:
    for c in input_str:
        if c.isalpha():
            continue
        if c in VALID_PUNCTUATION:
            continue
        if c.isspace():
            continue
        return False
    return True


def extract_pyphen_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    tokens = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", text, flags=re.UNICODE)
    return " ".join(tokens)


def count_syllables(text: str, lang: str) -> int:
    if not text:
        return 0

    if lang in {"zh", "yue"}:
        return len(re.findall(r"[\u4e00-\u9fff]", text))

    if lang == "ko":
        return len(re.findall(r"[\uac00-\ud7a3]", text))

    if lang == "th":
        clean_text = "".join(
            ch
            for ch in text
            if unicodedata.category(ch)[0] in {"L", "M"} or ch.isspace()
        )
        return len(syllable_tokenize(clean_text))

    if lang == "ja":
        _, count = ja_split_syllables(text)
        return count

    if lang == "vi":
        text = unicodedata.normalize("NFKC", text)
        tokens = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", text, flags=re.UNICODE)
        return len(tokens)

    clean_text = extract_pyphen_text(text)
    if not clean_text:
        return 0

    pyphen_lang = PYPHEN_LANG_MAP.get(lang, "en_US")
    if pyphen_lang not in _PYPHEN_CACHE:
        try:
            _PYPHEN_CACHE[pyphen_lang] = pyphen.Pyphen(lang=pyphen_lang)
        except Exception:
            if "en_US" not in _PYPHEN_CACHE:
                _PYPHEN_CACHE["en_US"] = pyphen.Pyphen(lang="en_US")
            pyphen_lang = "en_US"

    dic = _PYPHEN_CACHE[pyphen_lang]
    total = 0
    for word in clean_text.split():
        if word:
            total += len(dic.inserted(word).split("-"))
    return total


def map_to_class(speed: float, delta: float = 0.25) -> float:
    return round(speed / delta) * delta


def read_all_metadata(input_dir: str):
    input_path = Path(input_dir) / "csv_train"
    csv_files = list(input_path.glob("metadata_*_full.csv"))

    if not csv_files:
        print(f"No proper csv files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} metadata files: {[f.name for f in csv_files]}")
    all_tasks = []
    for csv_file in csv_files:
        lang_code = csv_file.stem.split("_")[1]
        print(f"lang_code: {lang_code}")
        all_tasks.append((lang_code, csv_file))
    return all_tasks


def read_csv_file(csv_path: Path):
    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3:
                continue
            audio_path, duration, text = parts
            try:
                duration = float(duration)
            except ValueError:
                continue
            items.append((audio_path, text, duration))
    return items


def build_sample(audio_path: str, text: str, duration: float, lang_code: str):
    if duration <= 0 or not check_valid_chars(text):
        return None

    syllable_count = count_syllables(text, lang_code)
    if syllable_count <= 0:
        return None

    return {
        "audio_path": audio_path,
        "text": text,
        "duration": duration,
        "speed_syllables": map_to_class(syllable_count / duration),
        "lang": lang_code,
    }


def select_split_items(raw_items, lang_code: str, rng: random.Random):
    shuffled_items = list(raw_items)
    rng.shuffle(shuffled_items)

    train_results = []
    val_results = []
    train_durations = []
    val_durations = []
    train_duration_sum = 0.0

    for audio_path, text, duration in shuffled_items:
        sample = build_sample(audio_path, text, duration, lang_code)
        if sample is None:
            continue

        if len(val_results) < VAL_SAMPLES_PER_LANG:
            val_results.append(sample)
            val_durations.append(duration)
            continue

        if train_duration_sum < TRAIN_HOURS_PER_LANG * 3600:
            train_results.append(sample)
            train_durations.append(duration)
            train_duration_sum += duration

        if len(val_results) >= VAL_SAMPLES_PER_LANG and train_duration_sum >= TRAIN_HOURS_PER_LANG * 3600:
            break

    return train_results, train_durations, val_results, val_durations


def write_arrow(output_path: Path, rows):
    with ArrowWriter(path=output_path.as_posix(), writer_batch_size=10000) as writer:
        for row in rows:
            writer.write(row)
        writer.finalize()


def write_duration_json(output_path: Path, durations):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False)


def write_speed_syllables_stats(out_dir: Path, split_name: str, rows, write_histogram: bool):
    counter = Counter(row["speed_syllables"] for row in rows)
    sorted_items = sorted(counter.items(), key=lambda x: x[0])
    counts = {str(k): v for k, v in sorted_items}

    with open(
        out_dir / f"speed_syllables_counts_{split_name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)

    if write_histogram:
        plt.figure(figsize=(12, 5))
        plt.bar([str(k) for k, _ in sorted_items], [v for _, v in sorted_items])
        plt.xlabel("speed_syllables")
        plt.ylabel("count")
        plt.title(f"speed_syllables histogram ({split_name})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / f"speed_syllables_hist_{split_name}.png", dpi=200)
        plt.close()


def prepare_all(inp_dir: str, out_dir_root: str, dataset_name: str, seed: int = 42):
    inp_dir = Path(inp_dir)
    out_dir_root = Path(out_dir_root)
    out_dir = Path(f"{out_dir_root / dataset_name}_srp")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Will be saved to {out_dir}")

    tasks = read_all_metadata(inp_dir)

    train_rows = []
    val_rows = []
    train_durations = []
    val_durations = []

    for idx, (lang_code, csv_path) in enumerate(tasks):
        print(f"\nProcessing Language: {lang_code} (from {csv_path.name})")
        raw_items = read_csv_file(csv_path)
        print(f"Loaded {len(raw_items)} rows before filtering.")

        rng = random.Random(seed + idx)
        lang_train, lang_train_durations, lang_val, lang_val_durations = select_split_items(
            raw_items, lang_code, rng
        )

        train_rows.extend(lang_train)
        train_durations.extend(lang_train_durations)
        val_rows.extend(lang_val)
        val_durations.extend(lang_val_durations)

        print(
            f"Selected train={len(lang_train)} samples ({sum(lang_train_durations) / 3600:.2f}h), "
            f"val={len(lang_val)} samples"
        )

    write_arrow(out_dir / "raw.arrow", train_rows)
    write_arrow(out_dir / "raw_val.arrow", val_rows)
    write_duration_json(out_dir / "duration.json", train_durations)
    write_duration_json(out_dir / "duration_val.json", val_durations)
    write_speed_syllables_stats(out_dir, "train", train_rows, write_histogram=True)

    print("\n" + "=" * 50)
    print(f"Train samples: {len(train_rows)}")
    print(f"Train hours:   {sum(train_durations) / 3600:.2f}")
    print(f"Val samples:   {len(val_rows)}")
    print(f"Val hours:     {sum(val_durations) / 3600:.2f}")
    print(f"Saved to:      {out_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inp_dir",
        type=str,
        default="/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/datasets",
        help="Root dir containing csv_train/metadata_*_full.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/F5-TTS/data",
        help="Output root dir for SRP data",
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_all(args.inp_dir, args.out_dir, args.dataset_name, args.seed)


if __name__ == "__main__":
    main()
