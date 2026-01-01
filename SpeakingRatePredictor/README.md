# Speaking Rate Predictor
Speaking Rate Predictor for [Cross-Lingual F5-TTS](https://arxiv.org/abs/2509.14579)

## Installation
This module is built upon the **F5-TTS** environment. 

1. Ensure you have the F5-TTS environment set up (refer to the main repository [F5-TTS repo](https://github.com/SWivid/F5-TTS/)).
2. Install the additional dependency:

```bash
pip install pyphen
```

## Data Augmentation: Silence Injection
During training, silence is randomly inserted into audio samples to improve generalization. The target speed label remains unchanged.

- **Duration**: 30% ~ 70% of the original sample length.
- **Mode Probabilities**:
  - `None` (40%): No silence added.
  - `Front` (20%): Silence added to the start.
  - `Back` (20%): Silence added to the end.
  - `Both` (20%): Silence added to both start and end.