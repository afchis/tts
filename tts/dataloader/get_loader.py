import os
import random

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from .datasets import LJSpeechDataset


def _collate_fn(batch):
    mel_spec, phoneme = list(), list()
    for _mel_spec, _phoneme in batch:
        mel_spec.append(_mel_spec)
        phoneme.append(_phoneme)
    # mel_len = max([_mel_spec.size(1) for _mel_spec in mel_spec])
    # pho_len = max([_phoneme.size(0) for _phoneme in phoneme])
    # mel_spec = [F.pad(_mel_spec, (0, mel_len-_mel_spec.size(1)), "constant", 0) for _mel_spec in mel_spec]
    # phoneme = [F.pad(_phoneme, (0, pho_len-_phoneme.size(0)), "constant", 0) for _phoneme in phoneme]
    # mel_spec = torch.stack(mel_spec)
    # phoneme = torch.stack(phoneme)
    return mel_spec, phoneme


def _collate_fn_vox_train(batch):
    speakers = list()
    if batch[0][0].size(0) == 1:
        raise NotImplementedError("dec _collate_fx_vox for data type: wav")
    else:
        mel_len = random.randint(140, 180)
        for _speaker in batch:
            utterances = list()
            for _utterance in _speaker:
                while _utterance.size(0) < mel_len:
                    _utterance = torch.cat([_utterance, _utterance])
                start = int(random.random()*(_utterance.size(0)-mel_len-1))
                _utterance = _utterance[start:start+mel_len]
                utterances.append(_utterance)
            utterances = torch.stack(utterances)
            speakers.append(utterances)
    return torch.stack(speakers)


def _collate_fn_vox_test(batch):
    raise NotImplementedError("VOX2 dataset collate_fn")


def get_loader(params, stage):
    if params["dataset"]["name"] == "LJSpeech":
        dataset = LJSpeechDataset(params, stage)
        collate_fn = _collate_fn
    else:
        raise NotImplementedError("datasets: []")
    # elif params["dataset"]["name"] == "VOX2":
    #     dataset = VOX2(params, stage)
    #     if stage == "train":
    #         collate_fn = _collate_fn_vox_train
    #     else:
    #         collate_fn = _collate_fn_vox_test
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    sampler = (torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    ) if world_size > 1 else None)
    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        num_workers=0 if world_size > 1 else params["num_workers"],
        shuffle=True if stage == "train" and world_size == 1 else False,
        sampler=sampler,
        collate_fn=collate_fn,
    )
    return dataloader


if __name__ == "__main__":
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    params = {
        "dataset": {
            "name": "VOX2",
            "num_utterance": 10,
            "wav_len": 16000,
            "data_type": "mel"
        },
        "batch_size": 8,
        "num_workers": 1
    }
    loader = get_loader(params=params,
                        stage="train")
    wavs = next(iter(loader))
    print(wavs.shape)

