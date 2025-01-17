import os
from threadpoolctl import threadpool_limits

import torch
import torchaudio
from torchaudio import transforms as T

import numpy as np
import librosa
import matplotlib
import matplotlib.pyplot as plt


class GetSpec:
    def __init__(self, wav_sample, mode="mel"):
        self.mode = mode
        _, self.sample_rate = torchaudio.load(wav_sample)
        self.get_spec = self._get_spec()

    def _get_spec(self):
        if self.mode == "mel":
            transform = T.MelSpectrogram(sample_rate=self.sample_rate,
                                         n_fft=2000,
                                         win_length=400,
                                         hop_length=400,
                                         n_mels=80)
        elif self.mode == "spec":
            raise NotImplementedError # transform = T.Spectrogram()
        else:
            raise Exception("Wrong mode for get Spectogram.")
        return transform

    def __call__(self, data_path):
        wav, sr = torchaudio.load(data_path)
        assert sr == self.sample_rate, f"The file `{data_path}` has a different sample rate. Target sample rate is {self.sample_rate}."
        return self.get_spec(wav)


class Librosa:
    def __init__(self, wav_sample, frame_size=50, hop_size=10):
        self.normalized = False
        _, self.sample_rate = librosa.load(wav_sample)
        self.frame_size = int(self.sample_rate * frame_size / 1000)
        self.hop_size = int(self.sample_rate * hop_size / 1000)

    def norm_spec(self, spec):
        self.normalized = True
        return (spec + 100.) / 100

    def denorm_spec(self, spec):
        return (spec * 100) - 100.
    
    def power_to_db(self, spec):
        if self.normalized == True:
            return librosa.power_to_db(spec)
        else:
            return spec

    def get_wav(self, wav_path, norm_and_filt=False):
        wav, self.sample_rate = librosa.load(wav_path)
        if norm_and_filt:
            wav = librosa.util.normalize(wav)
            wav = librosa.effects.preemphasis(wav)
        return wav

    def get_stft(self, wav, plot=False, log_scale=False):
        spec = librosa.stft(wav, n_fft=self.frame_size, hop_length=self.hop_size)
        if plot:
            y_axis = "log" if log_scale else "linear"
            fig, ax = plt.subplots()
            img = librosa.display.specshow(spec, hop_length=self.hop_size, x_axis="time", y_axis=y_axis, ax=ax)
            fig.colorbar(img, ax=ax, format="%+2.f")
            ax.set(title="STFT spectrogram display")
        return spec

    def get_mel(self, wav, n_mels, log_compression=False, plot=False):
        mel = librosa.feature.melspectrogram(y=wav, sr=self.sample_rate, n_fft=self.frame_size, hop_length=self.hop_size, n_mels=n_mels)
        if plot:
            # mel = librosa.power_to_db(mel, ref=np.max, top_db=100.)
            fig, ax = plt.subplots()
            img = librosa.display.specshow(mel, x_axis="time", y_axis="mel", sr=self.sample_rate, fmax=int(self.sample_rate/2), ax=ax)
            fig.colorbar(img, ax=ax, format="%+2.f")
            ax.set(title="Mel spectrogram display")
        return mel

    def plot_pred(self, mels, epoch, _iter, output_path, num_preds=4):
        if len(mels["target"]) < num_preds: num_preds = len(mels["target"])
        mels["pred_decoder"] /= mels["pred_decoder"].max() # TEMP
        mels["pred_post"] /= mels["pred_post"].max() # TEMP
        nrows = len(mels)
        fig, ax = plt.subplots(nrows=nrows, ncols=num_preds, sharex=True, figsize=(30, 20))
        for i, (key, value) in enumerate(mels.items()):
            for j in range(num_preds):
                img = librosa.display.specshow(
                    self.denorm_spec(value[j].numpy()),
                    x_axis="time",
                    y_axis="mel",
                    sr=self.sample_rate,
                    fmax=int(self.sample_rate/2),
                    ax=ax[i] if num_preds == 1 else ax[i, j]
                )
                if num_preds == 1:
                    ax[i].set(title=key)
                else:
                    ax[i, j].set(title=key)
        fig.colorbar(img, ax=ax, format="%.1f")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(output_path+f"/e{epoch:03d}_i{_iter:06d}.png", dpi=400)
        matplotlib.pyplot.close()

    def get_f0(self, wav, plot=False, spec=None, log_scale=False):
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=wav,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=self.sample_rate,
            frame_length=self.frame_size,
            hop_length=self.hop_size
        )
        times = librosa.times_like(f0)
        if plot:
            y_axis = "log" if log_scale else "linear"
            fig, ax = plt.subplots()
            img = librosa.display.specshow(spec, x_axis="time", y_axis=y_axis, ax=ax)
            ax.set(title="pYIN fundamendal frequency esrimation")
            ax.plot(times, f0, label='f0', color="cyan", linewidth=3)
            ax.legend(loc="upper right")
        return f0

    def __call__(self, data_path):
        wav, sr = librosa.load(data_path)
        raise NotImplementedError


