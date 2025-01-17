import os
import glob
import csv
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from .utils import GetSpec, Librosa, text_to_sequence


class VOX2(Dataset):
    _data_path_ = "/storage/prj/ProtopopovI/db/audio/vox2/dev/aac/"
    _sample_rate_ = 16000

    def __init__(self, params, stage="train"):
        super().__init__()
        self.wav_len = params["dataset"].get("wav_len")
        self.num_utterance = params["dataset"]["num_utterance"]
        self.data_type = params["dataset"]["data_type"]
        self.stage = stage
        self.speaker_names = self._prepare_data()
        wav_sample = glob.glob(os.path.join(self._data_path_, self.speaker_names[0], "*", "*VAD.wav"))[0]
        self.librosa = Librosa(wav_sample=wav_sample)
        self.torchaudio_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._sample_rate_,
            hop_length=int(self._sample_rate_ * 10. / 1000),
            n_fft=int(self._sample_rate_ * 25. / 1000),
            f_min=50,
            n_mels=80
        )
    
    def __len__(self):
        return len(self.speaker_names)

    def _prepare_data(self):
        speaker_names = sorted(os.listdir(self._data_path_))
        segm_len = len(speaker_names) // 12
        if self.stage == "train":
            speaker_names = speaker_names[:10 * segm_len]
        elif self.stage == "valid":
            speaker_names = speaker_names[10 * segm_len: 11 * segm_len]
        elif self.stage == "test":
            speaker_names = speaker_names[11 * segm_len:]
        else:
            raise Exception("Wrong mode for VOX2 dataset")
        return speaker_names
    
    def _get_audios(self, file_names):
        if self.data_type == "mel":
            mel_specs = list()
            for file_path in file_names:
                mel_path = file_path[:-7] + ".npy"
                if os.path.exists(mel_path):
                    mel_spec = torch.from_numpy(np.load(mel_path))
                    mel_specs.append(mel_spec)
                else:
                    wav = torchaudio.load(file_path)[0]
                    mel_spec = self.torchaudio_mel(wav / torch.abs(wav).max()).squeeze().T
                    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-9))
                    np.save(mel_path, mel_spec)
                    mel_specs.append(mel_spec)
            return mel_specs
        else:
            wavs = [torchaudio.load(wav_path)[0] for wav_path in file_names]
            wavs = [wav / torch.abs(wav).max() for wav in wavs]
            return wavs
    
    def __getitem__(self, idx):
        speaker_name = self.speaker_names[idx]
        file_names = glob.glob(os.path.join(self._data_path_, speaker_name, "*", "*VAD.wav"))
        file_names = random.sample(file_names, self.num_utterance)
        utterance = self._get_audios(file_names)
        return utterance


if __name__ == "__main__":
    params = {
        "dataset": {
            "name": "VOX2",
            "data_type": "mel",
            "num_utterance": 10,
            "wav_len": 16000
        },
    }
    dataset = VOX2(params=params, stage="train")
    for i in range(len(dataset)):
        print("train:", i)
        wavs = dataset[i]
    dataset = VOX2(params=params, stage="valid")
    for i in range(len(dataset)):
        print("valid:", i)
        wavs = dataset[i]
    dataset = VOX2(params=params, stage="test")
    for i in range(len(dataset)):
        print("test:", i)
        wavs = dataset[i]
        # print(wavs.shape)
    # wavs = dataset[0]
    # print(wavs.shape)

