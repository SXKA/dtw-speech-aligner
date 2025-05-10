# dtw-speech-aligner
A Python tool for speech segment alignment using Dynamic Time Warping (DTW). 
Assumes input audio has been pre‑processed by VAD (voice activity detection).
## Features
- Subsequence DTW alignment of two speech segments (query vs. reference) 
- Supports MFCC and/or fundamental‑frequency (F0) features  
- Outputs clipped reference audio and optional diagnostic plots
## Usage
```bash
python main.py \
  --query_path   path/to/query_audio \
  --reference_path path/to/reference_audio \
  [--feat_types mfcc f0] \
  [--save_plot]
```
- `query_path`: path to the (pre‑VAD) query audio.
- `reference_path`: path to the (pre‑VAD) reference audio.
- `feat_types`: which features to use: mfcc, f0 (default: mfcc).
- `save_plot`: save DTW & spectrogram plots.
## Examples
### Chinese
#### Audio
Query (TTS): [query_chinese.wav](https://github.com/SXKA/dtw-speech-aligner/blob/main/audio/query_chinese.wav)

Reference (Human): [reference_chinese.wav](https://github.com/SXKA/dtw-speech-aligner/blob/main/audio/reference_chinese.wav)

Clipped segment (DTW): [clip_chinese.wav](https://github.com/SXKA/dtw-speech-aligner/blob/main/audio/clip_chinese.wav)
#### Visualization
##### Alignment
<div align="center">
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/alignment_chinese.png" alt="alignment_chinese"/>
</div>

##### Mel-spectrogram & f0
<div align="center">
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/f0_mel_spec_query_chinese.png" alt="f0_mel_spec_query_chinese"/>
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/f0_mel_spec_reference_chinese.png" alt="f0_mel_spec_reference_chinese"/>
</div>

##### MFCC DTW paths
<div align="center">
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/mfcc_dtw_chinese.png" alt="mfcc_dtw_chinese"/>
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/mfcc_delta_dtw_chinese.png" alt="mfcc_delta_dtw_chinese"/>
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/mfcc_delta_delta_dtw_chinese.png" alt="mfcc_delta_delta_dtw_chinese"/>
</div>

### English
#### Audio
Query (TTS): [query_english.wav](https://github.com/SXKA/dtw-speech-aligner/blob/main/audio/query_english.wav)

Reference (Human): [reference_english.wav](https://github.com/SXKA/dtw-speech-aligner/blob/main/audio/reference_english.wav)

Clipped segment (DTW): [clip_english.wav](https://github.com/SXKA/dtw-speech-aligner/blob/main/audio/clip_english.wav)
#### Visualization
##### Alignment
<div align="center">
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/alignment_english.png" alt="alignment_english"/>
</div>

##### Mel-spectrogram & f0
<div align="center">
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/f0_mel_spec_query_english.png" alt="f0_mel_spec_query_english"/>
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/f0_mel_spec_reference_english.png" alt="f0_mel_spec_reference_english"/>
</div>

##### MFCC DTW paths
<div align="center">
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/mfcc_dtw_english.png" alt="mfcc_dtw_english"/>
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/mfcc_delta_dtw_english.png" alt="mfcc_delta_dtw_english"/>
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/mfcc_delta_delta_dtw_english.png" alt="mfcc_delta_delta_dtw_english"/>
</div>

### Taiwanese
#### Audio
Query (TTS): [query_taiwanese.wav](https://github.com/SXKA/dtw-speech-aligner/blob/main/audio/query_taiwanese.wav)

Reference (Human): [reference_taiwanese.mp3](https://github.com/SXKA/dtw-speech-aligner/blob/main/audio/reference_taiwanese.mp3)

Clipped segment (DTW): [clip_taiwanese.wav](https://github.com/SXKA/dtw-speech-aligner/blob/main/audio/clip_taiwanese.wav)
#### Visualization
##### Alignment
<div align="center">
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/alignment_taiwanese.png" alt="alignment_taiwanese"/>
</div>

##### Mel-spectrogram & f0
<div align="center">
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/f0_mel_spec_query_taiwanese.png" alt="f0_mel_spec_query_taiwanese"/>
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/f0_mel_spec_reference_taiwanese.png" alt="f0_mel_spec_reference_taiwanese"/>
</div>

##### MFCC DTW paths
<div align="center">
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/mfcc_dtw_taiwanese.png" alt="mfcc_dtw_taiwanese"/>
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/mfcc_delta_dtw_taiwanese.png" alt="mfcc_delta_dtw_taiwanese"/>
  <img src="https://github.com/SXKA/dtw-speech-aligner/blob/main/png/mfcc_delta_delta_dtw_taiwanese.png" alt="mfcc_delta_delta_dtw_taiwanese"/>
</div>
