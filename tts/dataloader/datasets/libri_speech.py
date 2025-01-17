import os
import glob
import csv
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from .utils import GetSpec, Librosa, text_to_sequence


class LibriSpeechDataset(Dataset):
    _glob_path_dict_ = {
        "small": "/dbs/audio/Libri/small/*/*/*.txt",
        "medium": "/dbs/audio/Libri/train-clean-100/*/*/*/*.txt",
        "large": "/dbs/audio/Libri/*/*/*/*/*.txt"
    }

    def __init__(self, data_size="medium"):
        super().__init__()
        self.anno_list = self._get_annotaions(self._glob_path_dict_[data_size])
        self.get_spec = GetSpec(wav_sample=self.anno_list[0][0])

    def _get_annotaions(self, data_paths):
        anno_list = list()
        file_txt_names = glob.glob(data_paths)
        for file_txt in file_txt_names:
            with open(file_txt) as f:
                lines = f.readlines()
                for wav_name, text in [line.split(" ", 1) for line in lines]:
                    wav_name = [wav_name + "-norm.wav"]
                    wav_path = "/".join(file_txt.split("/")[:-1] + wav_name)
                    anno_list.append((wav_path, text))
        return anno_list
        
    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, idx):
        wav_path, text = self.anno_list[idx]
        mel_spec = self.get_spec(wav_path).transpose(1, 0)
        phoneme = text_to_sequence(text=text,
                                   cleaner_names=["english_cleaners"])
        phoneme = torch.from_numpy(phoneme)
        return mel_spec, phoneme


if __name__ == "__main__":
    # dataset = LibriSpeechDataset(data_size="small")
    # dataset = LJSpeech(stage="train")
    # mel_spec, phoneme = dataset[0]
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


