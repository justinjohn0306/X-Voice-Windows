# SFT Modification Instructions (for Multilingual F5-TTS)
## 1. preprocess
1. 需要读取合成语音数据集和原始30000h语音数据集
2. result新增三个信息
   1. `total_text`: ref text + target text
   2. `prompt_path`: generated prompt得绝对路径
   3. `prompt_frames`: generated prompt的帧长
3. `duration.json`需要给prompt + target的总帧长
4. 生成的data根目录：`"{dataset_name}_{tokenizer}_sft`

## 2. config✅
1. 新增`model.sft`(bool)
   > [configs/F5TTS_v1_Base_multilingual_full_catada_sft.yaml#L26](src/f5_tts/configs/F5TTS_v1_Base_multilingual_full_catada_sft.yaml#L26)
2. 新增`model.use_total_text`(bool)，用来判断是只给target text还是给ref text + target text
   > [configs/F5TTS_v1_Base_multilingual_full_catada_sft.yaml#L27](src/f5_tts/configs/F5TTS_v1_Base_multilingual_full_catada_sft.yaml#L27)

## 3. model/dataset.py✅
1. `load_dataset`
   >[model/dataset.py#L365](src/f5_tts/model/dataset.py#L365)
   1. 新增参数`sft`
   2. 新增变量`prompt_frames`
   3. 根据sft的值，判断调用哪个`CustomDataset`
      1. `if not sft`, 继续用`CustomDataset`
      2. `if sft`
         1. `if prompt_frames is None` 抛错
         2. 否则用`CustomDataset_sft`（需要传参`prompt_frames`）
2. `CustomDataset_sft`
   >[model/dataset.py#L172](src/f5_tts/model/dataset.py#L172)
   1. `__init__`: 加`prompt_frames`
   2. `get_frame_len`: 传给`DynamicSampler`的需要是`prompt_len` + `len`
   3. `__getitem__`: 返回值增加`total_text`和`prompt_mel`
      1. `prompt_path = os.path.join(self.root_dir, prompt_path)`需要确认 `?`
3. `collate_fn_sft`: 返回值新增三个属性
   >[model/dataset.py#L481](src/f5_tts/model/dataset.py#L481)
   1. `total_text`
   2. `total_text_lengths`
   3. `prompt_mel_lengths`

## 4. model/utils.py✅
1. `get_tokenizer`: 增加参数：`sft`
   >[model/utils.py#L382](src/f5_tts/model/utils.py#L382)
2. `mask_from_prompt_lens` (`CFM_SFT`要用)
3. `prefix_text_padding`

## 5. train_sft.py✅
>[train/train_sft.py](src/f5_tts/train/train_sft.py)
1. 适配上面修改的接口
   1. `load_dataset`额外接收参数`sft`
   2. `get_tokenizer`额外接收参数`sft`
2. 需要确保`pretrained_path`一定存在
3. `CFM_SFT`
   1. `text_num_embeds` 需要 +1
   2. 额外接收参数`use_total_text`
4. `Trainer_SFT`
   1. 额外接收参数`use_total_text`

## 6. trainer_sft.py✅
>[model/trainer_sft.py](src/f5_tts/model/trainer_sft.py)
1. __init__.py
   1. 传`use_total_text`
2. `load_checkpoint`
   1. 如果`self.checkpoint_path`里没有`.pt`或`.safetensors`，从`pretrained_path`找ckpt
   2. 词表后面加prompt token
3. 改`collate_fn_sft`
4. 修改训练传参
   1. `text_inputs = batch["total_text"] if self.use_total_text else batch["text"]`
   2. `prompt_mel_lengths = batch["prompt_mel_lengths"]`
   3. `loss, cond, pred = self.model(mel_spec, text=text_inputs, lens=mel_lengths, prompt_lens=prompt_mel_lengths, noise_scheduler=self.noise_scheduler)`
5. sample示例
   1. `infer_text`加分支
   2. 改`sample`的duration传参

## 7. cfm_sft.py✅
>[model/cfm_sft.py](src/f5_tts/model/cfm_sft.py)
1. 修改导入`from f5_tts.model.utils import (mask_from_prompt_lens, prefix_text_padding)`
2. __init__需要额外接收`use_total_text`
3. `sample` & `forward`：当 `if not self.use_total_text` 时，需要补 `N * prompt token + ". "`（同 gp 的 PlanF）
   1. 文本结构变成：
      1. `[prompt_token x N] + [anchor token ids] + [target text token ids]`
   2. 这里的 `anchor token ids` 对应固定字符串 `". "` 编码后的 token id 序列
   3. 同时还需要构造一份和 text 对齐的 `language_ids`（二维 `[b, nt]`）
      1. `prompt_token` 对应位置：填 `-1`
         1. 含义：这些位置在 `TextEmbedding` 里不做语言融合
      2. `anchor ". "` 对应位置：填 unknown `lang_id`（当前约定 `id = 30`）
      3. 普通 `target text` 对应位置：填正常目标语种 id
   4. 也就是说，`cfm_sft.py` 不只负责改 `text`，还要负责把前缀部分对应的 per-token `language_ids` 一起构造好并传下去，变成
      1. `[-1 x N] + [unknown_lang_id x anchor_len] + [target_lang_id x target_text_len]`
4. `rand_span_mask = mask_from_prompt_lens(prompt_lens, lens)`

## 8. dit.py✅
>[model/backbones/dit.py](src/f5_tts/model/backbones/dit.py)
1. `TextEmbedding.__init__` 和 `DiT.__init__` 增加 `sft`
   1. `TextEmbedding` 通过 `self.sft` 控制当前走预训练分支还是 SFT 分支
2. `train_sft.py` 构建 `DiT` 时需要显式传 `sft=model_cfg.model.sft`
3. 原本的 `dit.py` 的 `forward` 仍然兼容下面三种输入的 `language_ids`
   1. `list[str]`
   2. `torch.Tensor with shape [b]`
   3. `torch.Tensor with shape [b, nt]`（`cfm_sft` 喂给 `dit.py` 这个）
4. `TextEmbedding` 里分成两套语言融合逻辑
   1. `if not self.sft`
      1. 继续走原来的普通语言融合逻辑
   2. `else`
      1. 走 SFT / PlanF 的语言融合逻辑
5. 在 `sft=True` 分支下：
   1. 输入的 `language_ids` 需要支持二维 `[b, nt]`，并允许出现特殊值 `-1`
   2. 语义约定：
      1. `language_ids == -1`
         1. 这些位置不做语言融合
         2. 对应前缀 `prompt_token`
      2. `language_ids != -1`
         1. 正常做语言融合
         2. 这里既包括：
            1. 前缀 `anchor ". "` 使用 unknown `lang_id = 30`
            2. 普通 `target text` 使用正常目标语种 id
   3. 注意：不能直接拿 `-1` 去查 `lang_embed`
      1. 需要先把 `-1` 替换成一个安全 index 再查 embedding
      2. 然后只在 `language_ids != -1` 的位置应用融合结果

## 9. model/__init__.py✅
>[model/__init__.py](src/f5_tts/model/__init__.py)
1. `CFM_SFT`
2. `Trainer_SFT`
