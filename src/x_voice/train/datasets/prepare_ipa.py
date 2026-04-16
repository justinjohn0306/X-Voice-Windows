import os
import sys
import argparse
import json
import shutil
import multiprocessing
import re
import regex
from pathlib import Path
from typing import List, Union, Pattern, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import logging

import torchaudio
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

from ipa_v3_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v3
from ipa_v6_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v6
import random
from x_voice.model.utils import str_to_list_ipa_v3, str_to_list_ipa_v6, get_ipa_id
logger = logging.getLogger("phonemizer")
logger.setLevel(logging.ERROR)
logger.propagate = False

sys.path.append(os.getcwd())

TOKENIZERS = {}
def get_tokenizer(lang_code, tokenizer):
    espeak_code = get_ipa_id(lang_code)
    if espeak_code not in TOKENIZERS:
        try:
            if tokenizer == "ipa_v3":
                TOKENIZERS[espeak_code] = PhonemizeTextTokenizer_v3(language=espeak_code)
            elif tokenizer == "ipa_v6":
                TOKENIZERS[espeak_code] = PhonemizeTextTokenizer_v6(language=espeak_code, with_stress=True)
        except RuntimeError as e:
            print(f"Error initializing espeak for {espeak_code}: {e}")
            return None
    return TOKENIZERS[espeak_code]



def process_batch(batch_data, lang_code, tokenizer_str, wav_base):
    """
    batch_data: list[(audio_path, text, duration)]
    """
    tokenizer = get_tokenizer(lang_code, tokenizer=tokenizer_str)
    if tokenizer is None:
        return []

    texts = [item[1] for item in batch_data]
    
    try:
        ipa_texts = [tokenizer([text.strip()]) for text in texts]
    except Exception as e:
        print(f"IPA conversion failed for batch in {lang_code}: {e}")
        return []

    results = []
    for (audio_path, _, duration), ipa_text in zip(batch_data, ipa_texts):
        if not ipa_text.strip():
            continue
        results.append({
            "audio_path": str(wav_base / audio_path),
            "text": ipa_text,     
            "duration": duration,
            "language_id": lang_code
        })
    return results

def read_all_metadata(input_dir):
    """
    Scan all metadata_*.csv files under the input directory.
    """
    input_path = Path(input_dir)/'csvs'
    # Match files such as metadata_zh.csv and metadata_en.csv.
    all_files = list(input_path.glob("metadata_*.csv"))
    csv_files = []
    for f in all_files:
        csv_files.append(f)
    
    if not csv_files:
        print(f"No proper csv files found in {input_dir}")
        sys.exit(1)
        
    print(f"Found {len(csv_files)} metadata files: {[f.name for f in csv_files]}")
    all_tasks = [] # (lang_code, file_path)
    
    for csv_file in csv_files:
        # Extract the language code from the file name, e.g. metadata_zh.csv -> zh.
        lang_code = csv_file.stem.split('_')[1]
        if True:
            print(f"lang_code:{lang_code}")
            all_tasks.append((lang_code, csv_file))
        
    return all_tasks

def preload_wav_paths(wav_root: Union[str, Path]) -> Set[str]:
    existing_wavs = set()
    print(f"\nPreloading wav path in {wav_root}")
    
    for wav_path in tqdm(wav_root.rglob("*"), desc="scanning audio files"):
        existing_wavs.add(str(wav_path.absolute()))
    return existing_wavs


def read_csv_file(csv_path, target_duration=None, dnsmos=None, audio_set=None, wav_base=None):
    items = []
    all_duration = 0
    # Step 1: collect all valid rows with durations into a temporary list.
    temp_valid_lines = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # Skip the header row.
        for line in f:
            parts = line.strip().split('|')
            audio_dir = wav_base / parts[0]
            abs_path = str(audio_dir.absolute())
            if audio_set is not None and abs_path not in audio_set:
                continue
            if len(parts) == 4:
                mos = float(parts[2])
                if dnsmos is not None and mos < dnsmos:
                    continue
                else: 
                    temp_valid_lines.append((parts[0], parts[3], float(parts[1])))
            elif len(parts) == 3:  # Valid path|duration|text row.
                # Store the raw path, text, and duration first.
                temp_valid_lines.append((parts[0], parts[2], float(parts[1])))
            elif len(parts) == 2:  # Row without a duration value.
                print("Warning: no duration. Check the metadata file")
    
    # Step 2: shuffle valid rows only when a target duration is provided.
    if target_duration is not None:
        random.shuffle(temp_valid_lines)  # Shuffle valid rows.
    
    # Step 3: accumulate duration until the target is reached.
    for path, text, duration in temp_valid_lines:
        # Stop once the requested duration budget has been exceeded.
        if target_duration is not None and all_duration > target_duration * 3600:
            break
        items.append((path, text, duration))
        all_duration += duration
    
    return items, all_duration/3600

def prepare_all(inp_dir, out_dir_root, tokenizer, dataset_name, num_workers=16, duration_map=None, dnsmos=None, check_exists=False):
    inp_dir = Path(inp_dir)
    out_dir_root = Path(out_dir_root)
    out_dir = out_dir_root / f"{dataset_name}_{tokenizer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Will be saved to {out_dir}")
    
    tasks = read_all_metadata(inp_dir) # List[(lang, csv_path)]
    
    raw_arrow_path = out_dir / "raw.arrow"
    writer = ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=10000)
    
    vocab_set = set()
    global_token_stats = {}
    total_samples = 0
    duration_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for lang_code, csv_path in tasks:
            print(f"\nProcessing Language: {lang_code} (from {csv_path.name})")
            lang_duration = None
            if duration_map is not None:
                lang_duration = duration_map.get(lang_code, None)
                if lang_duration is None:
                    lang_duration = duration_map.get("default", None)
                print(f"Will try to choose {lang_duration} hours for {lang_code}")
            
            audio_set = None
            wav_base = inp_dir / "wavs"
            if check_exists:
                wav_dir = wav_base / lang_code
                audio_set = preload_wav_paths(wav_dir)
            raw_items, return_duration = read_csv_file(csv_path, lang_duration, dnsmos, audio_set, wav_base)
            if not raw_items:
                continue
            fixed_items = []
            for p, t, d in raw_items:
                fixed_items.append((p, t, d))
                
            print(f"Loaded {len(fixed_items)} lines. Duration: {return_duration}. Starting G2P conversion.")
            main_tokenizer = get_tokenizer(lang_code, tokenizer=tokenizer)
            
            batch_size = 1000 # Each worker processes 1000 rows at a time.
            batches = [fixed_items[i:i + batch_size] for i in range(0, len(fixed_items), batch_size)]
            
            futures = [executor.submit(process_batch, batch, lang_code, tokenizer, wav_base) for batch in batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  -> {lang_code}"):
                batch_results = future.result()
                
                for res in batch_results:
                    writer.write(res)
                    duration_list.append(res['duration'])
                    total_samples += 1
                    
                    # Update vocabulary statistics.
                    if tokenizer == "ipa_v3":
                        tokens = str_to_list_ipa_v3(res['text']) 
                    elif tokenizer == "ipa_v6":
                        tokens = str_to_list_ipa_v6(res['text'])
                    if random.random() <0.000001:
                        tqdm.write(res['text'])
                    
                    for token in tokens:
                        if token not in global_token_stats:
                            global_token_stats[token] = {"count": 0, "langs": set()}
                        global_token_stats[token]["count"] += 1
                        global_token_stats[token]["langs"].add(lang_code)
                    vocab_set.update(tokens)
    writer.finalize()
    
    total_duration = sum(duration_list)
    with open(out_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({
            "duration": duration_list, 
            "total_hours": total_duration / 3600,
            "total_samples": total_samples
        }, f, ensure_ascii=False)
        
    # 6. Save the vocabulary files.
    stats_path = out_dir / "vocab_stats.txt"
    print(f"\nSaving detailed vocabulary statistics to {stats_path}...")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Token\tTotalCount\tNumLangs\tLanguages\n")
        sorted_stats = sorted(global_token_stats.items(), key=lambda item: item[1]['count'], reverse=True)
        for token, stats in sorted_stats:
            lang_str = ",".join(sorted(list(stats['langs']))) # Sort for stable output.
            f.write(f"{token}\t{stats['count']}\t{len(stats['langs'])}\t{lang_str}\n")
            
    vocab_path = out_dir / "vocab.txt"
    final_vocab = sorted(list(vocab_set))
    #if "_" not in final_vocab: final_vocab.append("_")
    
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("<pad>\n")
        for v in final_vocab:
            f.write(f"{v}\n")
            
    print("\n" + "="*50)
    print(f"Total Samples: {total_samples}")
    print(f"Total Hours:   {total_duration/3600:.2f}")
    print(f"Vocab Size:    {len(final_vocab)}")
    print(f"Saved to:      {out_dir}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser()
    support_tokenizer = ["ipa_v3", "ipa_v6"]
    parser.add_argument("--inp_dir", type=str, default="/inspire/hdd/project/embodied-multimodality/chenxie-25019/rixixu/datasets/x-voice",help="Root dir containing metadata_*.csv and wavs/")
    parser.add_argument("--out_dir", type=str, default="./data",help="Output root dir for raw.arrow")
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU workers")
    parser.add_argument("--tokenizer",type=str, choices=support_tokenizer, default="ipa_v6")
    parser.add_argument("--dataset_name",type=str, default="multilingual_qyl_test")
    parser.add_argument("--dnsmos", type=float, default=None, help="min value of dnsmos")
    parser.add_argument("--check_exists", action="store_true", help="Whether to check if the audio file exists before processing.")
    
    
    args = parser.parse_args()
    duration_map=None
    
    prepare_all(args.inp_dir, args.out_dir, args.tokenizer, args.dataset_name, args.workers, duration_map, args.dnsmos, args.check_exists)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
    
# python src/x_voice/train/datasets/prepare_ipa.py --tokenizer ipa_v6 --dataset_name multilingual_xrx_test2_phase1 --dnsmos 2.0 --check_exists
