import torch
from torch.utils.data import Dataset
from functools import partial
from torchvision import transforms
import numpy as np
import torch
import librosa
import torchaudio


class VoiceDataset(Dataset):

    def __init__(self, voice_paths, noise_paths, pipeline, num_noise_clips=1):
        self.voice_paths = voice_paths
        self.noise_paths = noise_paths
        self.pipeline = pipeline
        self.num_noise_clips = num_noise_clips

    def __len__(self):
        return len(self.voice_paths)

    def __getitem__(self, idx):
        return self.pipeline(dict(
            voice=self.voice_paths[idx],
            noise=np.random.choice(self.noise_paths, self.num_noise_clips),
        ))


def get_files_to_spectrogram_pipeline(config, augment):
    return transforms.Compose([
        partial(load_audio_files, sampling_rate=config['sampling_rate']),
        partial(
            pad_or_trim,
            length=config['audio_length'],
            deterministic=not augment
        ),
        partial(
            mix_audio,
            noise_max=config['noise_max'] if augment else 1.,
            noise_min=config['noise_min'] if augment else 1.,
        ),
        get_audio_to_spectrogram_pipeline(config),
    ])


def load_audio_files(paths, sampling_rate):
    return dict(
        voice=load_audio(paths['voice'], sampling_rate),
        noise=np.concatenate([
            load_audio(path, sampling_rate)
            for path in paths['noise']
        ]),
    )


def load_audio(path, sampling_rate=None):
    audio, _ = librosa.load(path, sr=sampling_rate)
    assert len(audio) > 0, f'Silent audio at {path}'
    audio, _ = librosa.effects.trim(audio, top_db=60)
    audio = np.trim_zeros(audio)
    return audio


def pad_or_trim(sample, length, deterministic=True):

    def pad_or_trim_clip(audio):
        if len(audio) > length:
            start = (
                np.random.randint(0, len(audio) - length)
                if not deterministic else 0
            )
            return audio[start : start + length]
        else:
            if deterministic or np.random.random() > 0.5:
                return np.pad(audio, (0, length - len(audio)), mode='constant')
            else:
                return np.pad(audio, (length - len(audio), 0), mode='constant')

    return {
        name: pad_or_trim_clip(audio)
        for name, audio in sample.items()
    }


def mix_audio(sample, noise_min=1.0, noise_max=1.0):
    voice_power = np.sum(sample['voice'] ** 2)
    noise_power = np.sum(sample['noise'] ** 2)
    noise_strength = (np.random.random() * (noise_max - noise_min)) + noise_min
    return dict(
        mixed=(
            sample['voice']
            + np.sqrt(voice_power / noise_power)
            * sample['noise'] * noise_strength
        ),
        voice=sample['voice'],
    )


def get_audio_to_spectrogram_pipeline(config):
    return transforms.Compose([
        to_tensor,
        partial(
            audio_to_stft,
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            win_length=config['win_length'],
        ),
        extract_magnitude_and_phase,
        normalize,
        to_dict,
    ])


def to_tensor(sample):
    return {
        name: torch.tensor(audio, dtype=torch.float32)
        for name, audio in sample.items()
    }


def audio_to_stft(sample, n_fft=512, hop_length=None, win_length=None):
    return {
        name: torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        for name, audio in sample.items()
    }


def extract_magnitude_and_phase(sample):
    return {
        name: torch.stack((
            torch.sqrt((stft ** 2).sum(dim=2)),
            torch.atan2(stft[..., 1], stft[..., 0])
        ))
        for name, stft in sample.items()
    }


def normalize(sample):
    output = dict(magnitude_scale=sample['mixed'][0].max())
    for name, spectrogram in sample.items():
        output[name] = torch.stack((
            spectrogram[0] / output['magnitude_scale'],
            spectrogram[1] / np.pi
        ))
    return output


def to_dict(sample):
    output = dict(
        features=sample['mixed'],
        magnitude_scale=sample['magnitude_scale']
    )
    if 'voice' in sample:
        output['targets'] = sample['voice'][0].unsqueeze(0)
        output['targets_phase'] = sample['voice'][1]
    return output


def get_spectrogram_to_audio_pipeline(config):
    return transforms.Compose([
        expand_dimensions,
        unnormalize,
        spectrogram_to_stft,
        partial(
            stft_to_audio,
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            win_length=config['win_length'],
        ),
    ])


def expand_dimensions(sample):
    return {
        name: (
            tensor.unsqueeze(0)
            if (tensor.ndim < 3 and '_scale' not in name)
            else tensor
        )
        for name, tensor in sample.items()
    }


def unnormalize(sample):
    return dict(
        magnitude=sample['magnitude'] * sample['magnitude_scale'],
        phase=sample['phase'] * np.pi,
    )


def spectrogram_to_stft(sample):
    return dict(
        stft=torch.stack((
            sample['magnitude'] * torch.cos(sample['phase']),
            sample['magnitude'] * torch.sin(sample['phase']),
        ), dim=3)
    )


def stft_to_audio(sample, n_fft, hop_length, win_length):
    return torchaudio.functional.istft(
        sample['stft'],
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
