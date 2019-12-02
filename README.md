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
| Multiple cameras
| Overlapping images

### Object Tracking
We implemented a tracking algorithm that follow objects across frames. It uses correlation filters for auto-correlation and cross-correlation in order to generalize better to new data:

    def correlation(x1, x2):
        signal_shape = x1.shape[-2:]
        x1 = torch.rfft(x1, 2)
        x2 = torch.rfft(x2, 2)
        corr = complex_multiply(x1, complex_conjugate(x2))
        return torch.irfft(corr, 2, signal_sizes=signal_shape)

[See more](https://github.com/Aiwizo/capability/blob/master/object_tracking/correlation.py)

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
Investigated semi supervised approaches for use in problem formulations with very little data. Initially we tried [mixup and mixmatch on mnist with 10 examples](https://github.com/Aiwizo/mnist) with good results out-of-the-box. We have since tried the method on many other problems as well but it is less useful where we already have lots of data. The implementation and usage is relatively simple:
- [mixup (tensorflow)](https://github.com/Aiwizo/capability/blob/master/semi_supervised/mixup.py)
- [mixmatch (tensorflow, extension of mixup)](https://github.com/Aiwizo/capability/blob/master/semi_supervised/mixmatch.py)
- [mixmatch (pytorch)]([mixmatch-pytorch](https://github.com/FelixAbrahamsson/mixmatch-pytorch)

### Generating climbing problems
Two of our colleagues are passionate about bouldering. They developed an algorithm that creates problems on a climbing board. The chosen holds are strongly dependent on each other and was modelled in a few different ways:

- Decoupled sampling by predicting the next hold using modified loss
- Predict full board using the [Gumbel-softmax trick](https://pytorch.org/docs/stable/distributions.html#relaxedonehotcategorical) and modified loss for steps in-between
- Hybrid variantional autoencoder with adverserial loss
- Variational autoencoder
- Generative adverserial network

Features like difficulty were also introduced to the model to steer what kind of problem would be created.

### Discrete relaxation
Much like the [Gumbel-softmax trick](https://pytorch.org/docs/stable/distributions.html#relaxedonehotcategorical) that tries to let us get gradients through a discrete transformation, there is the idea that we can replace the derivative of a discrete function during training and analyze what happens mathematically. This has already shown to be very useful in [sequential modelling](https://arxiv.org/pdf/1801.09797.pdf) and creating better [variational autoencoders](https://arxiv.org/pdf/1906.00446.pdf). We implemented a couple of versions of our own and [the original](https://github.com/Aiwizo/capability/blob/master/kaiser_step.py) to try on some simple problems.

### Exploration by uncertainty estimation in world model
Attempt at using Bayesian model uncertainty for chosing paths for exploration. We had used variational inference a number of times and exploration seems like a natural place for its use. The [repo is called labyrinth and can be found here](https://github.com/samedii/labyrinth). We are using classic VI to learn the world model and using the uncertainty as a reward when learning an exploring agent. It gave ok results and could solve the labyrinth but the world model was a little weak. It might give better results using the [Kaiser step](https://github.com/Aiwizo/capability/blob/master/kaiser_step.py) that came out since to encode the latent state of the world.
