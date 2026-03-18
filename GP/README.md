# Batch inference代码
## config
[需要改的参数](https://github.com/QingyuLiu0521/Multilingual_F5-TTS/blob/main/GP/src/configs/F5TTS_v1_Base_debug_Emilia_gp_inference.yaml#L10-L11)

```yaml
source:
  expname: F5TTS_v1_Base_debug_Emilia_pinyin
  ckptstep: 2
  frac_lengths_mask: [0.1, 0.4]
```

- source.expname: 预训练f5-tts的yaml
- source.ckptstep: 指定要用的ckpt的步数
- source.frac_lengths_mask: 用来指定target text相比于ref tezt 的长度范围
  - 这个参数可以直接删掉，因为原本的inferencer.py的逻辑是在推理的过程中才会确定target text的，这一步直接移到preprocess时去处理更合理

## [prepare_emilia_gp_inference.py](https://github.com/QingyuLiu0521/Multilingual_F5-TTS/blob/main/GP/src/train/datasets/prepare_emilia_gp_inference.py)
这是Emilia推理的preprocess脚本，可以作为参考；相比[prepare_emilia.py](https://github.com/SWivid/F5-TTS/blob/27e20fcf39fbce3f187a61d4a3e92065d2d9b947/src/f5_tts/train/datasets/prepare_emilia.py) 最大的区别就是多了相对路径：`"rel_path": rel_path`

给30000h的多语言数据集写预处理脚本的时候和[prepare_emilia_gp_inference.py](https://github.com/QingyuLiu0521/Multilingual_F5-TTS/blob/main/GP/src/train/datasets/prepare_emilia_gp_inference.py)保持接口一致，也要有`"rel_path": rel_path`，此外还需要：
- `target_text`: 从对应csv中抽取相同语言的text作为target text，但是需要保证target text的token数量是ref text的10%～40%，可以加一个阈值去限制最小tokens数量不能低于当前ref text的token数量
- `total_duration`: 直接跟原版F5一样，用utf8字节数之比去算target speech's duration, 然后记录ref speech's duration + target speech's duration, 需要喂给dynamic sampler去分batch

## [inference_gp.py](https://github.com/QingyuLiu0521/Multilingual_F5-TTS/blob/main/GP/src/train/inference_gp.py)
[inference_gp.py#L36-L42](https://github.com/QingyuLiu0521/Multilingual_F5-TTS/blob/main/GP/src/train/inference_gp.py#L36-L42)

```py
    cfg.model.arch.attn_backend = "flash_attn"
    cfg.model.arch.attn_mask_enabled = True
```
按找之前的逻辑必须开flash_attn和mask，因为dynamic sampler是根据audio prompt的长度去分batch的，但是实际上每个sample的长度是由prompt len + target len决定的，所以用现有逻辑，batch内的sample长度并不相等, 不开mask生成的音频质量会有问题
- 如果让预处理的时候就确定target text和整体的长度，这样就可以像默认config一样（attn_backend = torch; attn_mask_enabled = False）
- 但即使修改处理还是建议保持上面原本的配置（[inference_gp.py#L36-L42](https://github.com/QingyuLiu0521/Multilingual_F5-TTS/blob/main/GP/src/train/inference_gp.py#L36-L42)）; 训练时用torch，推理时用flash attn应该没有太大影响因为计算结果近似一致; 而且在Emilia和LibriTTS上实测用flash attn推理得到的sim还会比用torch的略高。

```py
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
```

## [cfm.py#L231-L401](https://github.com/QingyuLiu0521/Multilingual_F5-TTS/blob/main/GP/src/model/cfm.py#L231-L401)
我之前写的原版F5的`sample_reverse`，参考这个逻辑，在multilingual f5的cfm.py写一个类似逻辑的`sample_reverse`（audio prompt放后面，target放前面）

## [inferencer_gp.py](https://github.com/QingyuLiu0521/Multilingual_F5-TTS/blob/main/GP/src/model/inferencer_gp.py)
这是batch inference的核心逻辑, 为了处理之前说的电音问题，需要audio prompt放后面，target放前面
- [inferencer_gp.py#L153-L155](https://github.com/QingyuLiu0521/Multilingual_F5-TTS/blob/main/GP/src/model/inferencer_gp.py#L153-L155)就直接把上面preprocess说的target text拿过来就行
- [inferencer_gp.py#L176-L185](https://github.com/QingyuLiu0521/Multilingual_F5-TTS/blob/main/GP/src/model/inferencer_gp.py#L176-L185)换成rixi写的multilingual f5的接口
- 其他逻辑不要变，变了会影响到后面二阶段的训练