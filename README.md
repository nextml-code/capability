# Introduction
The purpose of this repository is to prove our capability to deliver machine learning solutions. It's short and to the point.

- **Simplicity:** Simple solutions are usually good and "cool" tricks redundant. The goal is to show you that we are professionals who deliver value. We don't want to "impress" by making our projects seem more complicated than they are.
- **Cutting Edge:** With that said, we still want to prove that we have the skills (both in machine learning and programming) to develop and implement cutting-edge algorithms.
- **Sharing codee:** We can't share the entire codebases of our projects, but we can and will share relevant code-snippets.

# Projects

### Railroad Inspection

In collaboration with two companies working with railroad inspection and maintenance, we are developing algorithms for identifying critical infrastructure and potential damages. The data are images taken from trains.

One algorithm is responsible for detecting clamps on contact wires. It's a simple object detection problem, but there are some interesting challenges:

| Challenge | Solution |
|:----------|:---------|
| The dataset is very unbalanced. 99% of the images does not contains clamps, and there are different types of clamps | Sampeling images based on class-distribution, and then based on loss.|
| The algorithm has to analyze a massive amount of data, so speed is critical. |  A [custom architecture](https://github.com/Aiwizo/capability/blob/master/railroad_inspection/architecture.py) for creating and processing masks compared to alternatives such as Unet. |


We are also using image inpainting to detect anomalies that might be damages.

### Object Tracking

-
- (Tracking) Correlation filters, fourier transform in 2d => complex conjugate

state of the art for object tracking using correlation filters to follow objects across frames.


### Audio Denoising

- Generated data by combining voice audio clips from random pocasts as well as the [Common Voice dataset](https://voice.mozilla.org/en/datasets) with noise audio clips from various sources ([Freesound](https://annotator.freesound.org/), [AudioSet](https://research.google.com/audioset/index.html)).
- Mix signals such that the [signal-to-noise ratio](https://en.wikipedia.org/wiki/Signal-to-noise_ratio) is close to 0dB, but not always 0
- STFT to convert waveform to spectogram, compute magnitude spectrogram and phase
- UNet with EfficientNet backbone to predict a soft mask
- Multiplty mask with magnitude spectrogram of the mixed audio
- Training procedure includes the following techniques
  * optimizing using Adam
  * cyclical learning rate with warmup and decay
  * gradient accumulation to increase effective batch size
  * mean squared error loss

A snapshot of our data pipeline can be found [here](https://github.com/Aiwizo/capability/tree/master/audio_denoising/data.py)!

### Rambot Legal

# Spare Time

### Semi-supervised

Tested different semi supervised approaches on mnist
- mixmatch

### Generating climbing problems
Two of our colleagues are passionate about bouldering. They developed an algorithm that


### Labyrint


[mixmatch-pytorch](https://github.com/FelixAbrahamsson/mixmatch-pytorch)
