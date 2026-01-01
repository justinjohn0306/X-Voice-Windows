import math
import os
import random
import string
import sys
from pathlib import Path
import pyphen

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

SRC_DIR = (Path(__file__) / "../..").resolve()  # 指向 src/
sys.path.insert(0, str(SRC_DIR))

from model.modules import MelSpec
from model.dataset import count

# padded to max length mel batch
def padded_mel_batch(ref_mels):
    max_mel_length = torch.LongTensor([mel.shape[-1] for mel in ref_mels]).amax()
    padded_ref_mels = []
    for mel in ref_mels:
        padded_ref_mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    padded_ref_mels = padded_ref_mels.permute(0, 2, 1)
    return padded_ref_mels

# seedtts testset metainfo: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_seedtts_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            gt_wav = os.path.join(os.path.dirname(metalst), "wavs", utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo


# librispeech test-clean metainfo: gen_utt, ref_txt, ref_wav, gen_txt, gen_wav
def get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        # ref_txt = ref_txt[0] + ref_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        # gen_txt = gen_txt[0] + gen_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
        gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")

        metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    return metainfo

# get prompts from metainfo containing: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_inference_prompt_sp(
    metainfo,
    target_sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    n_mel_channels=100,
    hop_length=256,
    mel_spec_type="vocos",
    target_rms=0.1,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=2,
    max_secs=30,
):
    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, gt_num_units, gt_durations = (
        [[] for _ in range(num_buckets)] for _ in range(6)
    )

    mel_spectrogram = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    for utt, prompt_text, prompt_wav, gt_text, gt_wav in tqdm(metainfo, desc="Processing prompts..."):
        # Audio
        ref_audio, ref_sr = torchaudio.load(prompt_wav)
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
        if ref_rms < target_rms:
            ref_audio = ref_audio * target_rms / ref_rms
        assert ref_audio.shape[-1] > 5000, f"Empty prompt wav: {prompt_wav}, or torchaudio backend issue."
        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio)
        
        # gt_duration
        gt_audio, gt_sr = torchaudio.load(gt_wav)
        gt_duration = gt_audio.shape[-1] / gt_sr
        
        ref_mel_len = ref_audio.shape[-1] // hop_length

        # to mel spectrogram
        ref_mel = mel_spectrogram(ref_audio)
        ref_mel = ref_mel.squeeze(0)
        
        # genrated text's num units
        gt_num_unit = count(gt_text)
        
        # deal with batch
        assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
        assert (
            min_tokens <= ref_mel_len <= max_tokens
        ), f"Audio {utt} has duration {ref_mel_len*hop_length//target_sample_rate}s out of range [{min_secs}, {max_secs}]."
        bucket_i = math.floor((ref_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets)

        utts[bucket_i].append(utt)
        ref_rms_list[bucket_i].append(ref_rms)
        ref_mels[bucket_i].append(ref_mel)
        ref_mel_lens[bucket_i].append(ref_mel_len)
        gt_num_units[bucket_i].append(gt_num_unit)
        gt_durations[bucket_i].append(gt_duration)

        batch_accum[bucket_i] += ref_mel_len

        if batch_accum[bucket_i] >= infer_batch_size:
            # print(f"\n{len(ref_mels[bucket_i][0][0])}\n{ref_mel_lens[bucket_i]}\n{total_mel_lens[bucket_i]}")
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    gt_num_units[bucket_i],
                    gt_durations[bucket_i],
                )
            )
            batch_accum[bucket_i] = 0
            (
                utts[bucket_i],
                ref_rms_list[bucket_i],
                ref_mels[bucket_i],
                ref_mel_lens[bucket_i],
                gt_num_units[bucket_i],
                gt_durations[bucket_i],
            ) = [], [], [], [], [], []

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    gt_num_units[bucket_i],
                    gt_durations[bucket_i],
                )
            )
    # not only leave easy work for last workers
    random.seed(666)
    random.shuffle(prompts_all)

    return prompts_all