import os
import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import GetSpec, Librosa, text_to_sequence


class LJSpeechDataset(Dataset):
    _data_path_ = os.path.join("/dbs", "audio", "LJSpeech-1.1")

    def __init__(self, params, stage="train"):
        super().__init__()
        self.params = params
        self.stage = stage
        anno_list = self._get_annotations()
        self.anno_list = self._split_dataset(anno_list)
        self.librosa = Librosa(wav_sample=self.anno_list[0][0])

    def _get_annotations(self):
        os.makedirs(os.path.join(self._data_path_, "mels"), exist_ok=True)
        anno_list = list()
        with open(os.path.join(self._data_path_, "metadata.csv")) as csv_file:
            reader = csv.reader(csv_file, delimiter="\n", quotechar="|")
            for row in reader:
                row = row[0].split("|")
                file_name, text = row[0], row[1]
                wav_path = os.path.join(self._data_path_, "wavs", file_name+".wav")
                mel_path = os.path.join(self._data_path_, "mels", file_name)
                anno_list.append([wav_path, mel_path, text])
        return anno_list

    def _split_dataset(self, anno_list):
        if self.params["dataset"].get("num_data"): # TEMP
            return anno_list[:self.params["dataset"]["num_data"]] # TEMP
        if self.stage == "test":
            anno_list = anno_list[12100:]
        elif self.stage == "valid":
            anno_list = anno_list[11000:12100]
        else:
            anno_list = anno_list[:11000]
        return anno_list

    def __len__(self):
        return len(self.anno_list)

    def _get_mel(self, wav_path, mel_path):
        # # if os.path.exists(mel_path+".npy"):
        #     mel_spec = np.load(mel_path+".npy")
        # else:
        #     os.makedirs(os.path.join(self._data_path_, "wavs"), exist_ok=True)
        wav = self.librosa.get_wav(wav_path, norm_and_filt=True)
        mel_spec = self.librosa.get_mel(wav, n_mels=80)
        mel_spec = self.librosa.power_to_db(mel_spec)
        mel_spec = self.librosa.norm_spec(mel_spec)
            # np.save(mel_path, mel_spec)
        return mel_spec

    def __getitem__(self, idx):
        wav_path, mel_path, text = self.anno_list[idx]
        mel_spec = self._get_mel(wav_path, mel_path)
        phoneme = text_to_sequence(text=text,
                                   cleaner_names=["english_cleaners"])
        mel_spec = torch.from_numpy(mel_spec).transpose(0, 1)
        phoneme = torch.from_numpy(phoneme).long()
        return mel_spec, phoneme


if __name__ == "__main__":
    params = {
        "dataset": {
            "name": "LJSpeech",
        },
    }
    dataset = LJSpeechDataset(stage="train")
    mel_spec, phoneme = dataset[0]

