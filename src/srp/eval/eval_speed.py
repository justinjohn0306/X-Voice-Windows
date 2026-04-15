import os

import argparse
import time
from importlib.resources import files

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from tqdm import tqdm
import json
from x_voice.infer.utils_infer import load_checkpoint

from srp.eval.utils_eval import (
    get_inference_prompt_sp,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
)
from srp.model import SpeedPredictor

accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"

use_ema = True
target_rms = 0.1

PACKAGE_ROOT = files("srp")
rel_path = str(PACKAGE_ROOT.joinpath("../.."))

def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-n", "--expname", default="SpeedPredict_Emilia")
    parser.add_argument("-c", "--ckptstep", default=1, type=int)
    parser.add_argument("-t", "--testset", default="ls_pc_test_clean")
    parser.add_argument("-e", "--allowederror", default=0.1, type=float)
    
    args = parser.parse_args()
    
    exp_name = args.expname
    ckpt_step = args.ckptstep
    testset = args.testset
    allowed_error = args.allowederror
    
    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    
    model_cfg = OmegaConf.load(str(PACKAGE_ROOT.joinpath(f"configs/{exp_name}.yaml")))
    model_arc = model_cfg.model.arch
    
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    mel_spec_kwargs = model_cfg.model.mel_spec
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft
    
    if testset == "ls_pc_test_clean":
        metalst = rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst"
        librispeech_test_clean_path = "/hpc_stor03/sjtu_home/qingyu.liu/F5-TTS-main_v104/LibriSpeech/test-clean"  # test-clean path
        metainfo = get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path)

    elif testset == "seedtts_test_zh":
        metalst = rel_path + "/data/seedtts_testset/zh/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    elif testset == "seedtts_test_en":
        metalst = rel_path + "/data/seedtts_testset/en/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)
    
    output_dir = (
        f"{rel_path}/"
        f"results/{exp_name}_{ckpt_step}/{testset}"
    )

    prompts_all = get_inference_prompt_sp(
        metainfo,
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
        target_rms=target_rms,
        mel_spec_type=mel_spec_type,
        infer_batch_size=infer_batch_size,
    )

    model = SpeedPredictor(
        mel_spec_kwargs=mel_spec_kwargs,
        arch_kwargs = model_arc
    ).to(device)
    
    ckpt_path = rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}.pt"
    if not os.path.exists(ckpt_path):
        print("Loading from self-organized training checkpoints rather than released pretrained.")
        ckpt_path = rel_path + f"/{model_cfg.ckpts.save_dir}/model_{ckpt_step}.pt"
    dtype = torch.float32
    # dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)
    
    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # start batch inference
    accelerator.wait_for_everyone()
    start = time.time()
    
    with accelerator.split_between_processes(prompts_all) as prompts:
        test_results = []
        correct_predictions = 0
        total_predictions = 0
        absolute_errors = []
        relative_errors = []
        
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_mels, ref_mel_lens, gt_num_units, gt_durations = prompt
            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
    
            # Inference
            with torch.inference_mode():
                pred_speed = model.predict_speed(
                    audio=ref_mels,
                    # audio=ref_mels.half(),
                    lens=ref_mel_lens,
                )
            
                for i, (pred_sp, gt_num, gt_du) in enumerate(zip(pred_speed, gt_num_units, gt_durations)):
                    pred_du = gt_num / (pred_sp.item())
                    
                    # 计算绝对误差和相对误差
                    abs_error = abs(pred_du - gt_du)
                    rel_error = abs_error / gt_du * 100  # 转换为百分比
                    
                    absolute_errors.append(abs_error)
                    relative_errors.append(rel_error)                    
                    
                    if abs(pred_du - gt_du) <= allowed_error * gt_du or abs(pred_du - gt_du) <= 1:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    test_results.append({
                        "utts": utts,
                        "gt_duration": gt_du,
                        "predicted_duration": pred_du,
                        "predicted_speed": pred_sp.item(),
                        "num_units": gt_num,
                        "absolute_error": abs_error,
                        "relative_error": rel_error,
                    })
        
        gathered_results = accelerator.gather_for_metrics(test_results)
        correct_predictions = accelerator.gather(torch.tensor([correct_predictions], device=device)).sum().item()
        total_predictions = accelerator.gather(torch.tensor([total_predictions], device=device)).sum().item()
        absolute_errors = accelerator.gather(torch.tensor(absolute_errors, device=device))
        relative_errors = accelerator.gather(torch.tensor(relative_errors, device=device))
        
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        mae = absolute_errors.mean().item()  # 平均绝对误差
        mre = relative_errors.mean().item()  # 平均相对误差
        
        # 只在主进程中写入完整结果
        if accelerator.is_main_process:
            print(f"Total Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
            print(f"Mean Absolute Error: {mae:.2f}")
            print(f"Mean Relative Error: {mre:.2f}%")
            
            output_file = f"{output_dir}/result.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                # 写入所有收集到的结果
                for result in gathered_results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                # 写入总体准确率统计
                summary = {
                    "summary": {
                        "accuracy": accuracy,
                        "correct": correct_predictions,
                        "total": total_predictions,
                        "mean_absolute_error": mae,
                        "mean_relative_error": mre,
                    }
                }
                f.write(json.dumps(summary) + "\n")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60 :.2f} minutes.")
    
if __name__ == "__main__":
    main()

'''
Example:
accelerate launch src/eval/eval_speed.py -n SpeedPredict_Base -c 1 -t "ls_pc_test_clean" -e 0.1
'''
