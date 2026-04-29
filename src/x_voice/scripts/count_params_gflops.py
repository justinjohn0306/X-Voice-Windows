import os
import sys


sys.path.append(os.getcwd())

import thop
import torch

from x_voice.model import CFM, DiT

language_list = ["bg","cs","da","de","el","en","es","et","fi","fr","hr","hu","id","it","ja","ko","lt","lv","mt","nl","pl","pt","ru","ro","sk","sl","sv","th","vi","zh"]
""" ~160M """
# transformer = DiT(dim = 768, depth = 18, heads = 12, ff_mult = 2, languages=language_list, text_infill_lang_type="ada", time_infill_lang_type="time_concat", lang_dim=512, lang_dim_in_t=512, share_lang_embed=True)
# transformer = DiT(dim = 768, depth = 18, heads = 12, ff_mult = 2, text_dim = 512, conv_layers = 4, languages=language_list, text_infill_lang_type="ada", time_infill_lang_type="time_concat", lang_dim=512, lang_dim_in_t=512, share_lang_embed=True)
# transformer = DiT(dim = 768, depth = 18, heads = 12, ff_mult = 2, text_dim = 512, conv_layers = 4, long_skip_connection = True, languages=language_list, text_infill_lang_type="ada", time_infill_lang_type="time_concat", lang_dim=512, lang_dim_in_t=512, share_lang_embed=True)

""" ~340M """
# FLOPs: 364.4 G, Params: 338.9 M
# transformer = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4, languages=language_list, text_infill_lang_type="ada", time_infill_lang_type="time_concat", lang_dim=512, lang_dim_in_t=512, share_lang_embed=True)


model = CFM(transformer=transformer)
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
duration = 20
frame_length = int(duration * target_sample_rate / hop_length)
text_length = 150
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp, text):
        return self.model(inp, text, language_ids=["zh"])

wrapped_model = ModelWrapper(model)
flops, params = thop.profile(
    wrapped_model, 
    inputs=(torch.randn(1, frame_length, n_mel_channels), torch.zeros(1, text_length, dtype=torch.long))
)
print(f"FLOPs: {flops / 1e9} G")
print(f"Params: {params / 1e6} M")
