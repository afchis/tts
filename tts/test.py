import os
import json
import argparse

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tts.dataloader.get_loader import get_loader
from tts.models.get_model import get_model
from tts.utils.loss_metric import Criterions
from tts.utils.logger import TestLogger


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=-1)
parser_args = parser.parse_args()


class TesterMultiGPU:
    def __init__(self, test_params):
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.params = params
        if parser_args.gpu != -1:
            self.params["target_device"] = f"cuda:{parser_args.gpu}"
        if self.params["target_device"] == "cpu":
            self.device = "cpu"
            print(f"Initialization Tester: CPU", flush=True)
        else:
            self.device = self.rank
            if self.world_size == 1:
                print(f"Initialization Tester: GPU -> device: cuda:{self.rank}", flush=True)
            else:
                print(f"Initialization Tester: MultiGPU -> world_size: {self.world_size}, rank: {self.rank}]", flush=True)
        model = get_model(self.params).to(self.device)
        self._get_test_step()
        if self.world_size > 1:
            self.model = DDP(model, device_ids=[self.device])
        else:
            self.model = model
        self.load_model()
        self.test_loader = get_loader(self.params, stage="test")
        self.test_len = len(self.test_loader)
        self.librosa = self.test_loader.dataset.librosa
        self.criterion = Criterions(self.device)
        self.logger = TestLogger(tester=self)

    def _get_test_step(self):
        if self.params["network_name"] == "tacotron2":
            self.test_step = self._test_step_tacotron2
        else:
            raise NotImplementedError

    def load_model(self):
        chk_path = (".", "checkpoints", self.model.__class__.__name__)
        chk_path = os.path.join(*chk_path)
        chk_name = self.params["checkpoint_name"]
        if chk_name in os.listdir(chk_path):
            model_weights = os.path.join(chk_path, chk_name)
            self.model.load_state_dict(torch.load(model_weights))
            print("Checkpoint loaded:", model_weights)

    @staticmethod
    def setup_single():
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        print("Setup for train on cpu or single gpu: Done.", flush=True)

    @staticmethod
    def setup(rank, world_size, params):
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        print("rank:", os.environ["RANK"])
        # init_method = "tcp://" + params["uuid"] + "_0:1111" # for platform learning
        init_method = "tcp://localhost:11223"  # for local testing
        dist.init_process_group(backend="nccl", init_method=init_method,
                                rank=rank, world_size=world_size)
        print("Setup for Distributed Data Parallel: Done.", flush=True)

    def _data_to_device(self, data):
        data_to = list()
        for _data in data:
            data_to.append([item.to(self.device) for item in _data])
        return data_to

    def test(self):
        self.model.eval()
        self.model.inference = True
        pred_mels = list()
        target_mels = list()
        with torch.no_grad():
            for iter_, data in enumerate(self.test_loader):
                iter_ += 1
                test_out = self.test_step(data)
                pred_mels.append(test_out["mels"]["pred"])
                target_mels.append(test_out["mels"]["target"])
                print(iter_)
                if len(pred_mels) == 2:
                    self.librosa.plot_pred(pred_mels, target_mels, epoch=0, iter=iter_, output_path="./logs/test_imgs", num_preds=2)
                    pred_mels = list()
                    target_mels = list()
                # if self.rank == 0: self.logger.step(data=test_out)

    ########### tacotron steps ###########
    def _test_step_tacotron2(self, data):
        data = self._data_to_device(data)
        mel_spec, phoneme = data
        pred_mel = self.model(text_or_phoneme=phoneme)
        pred_mel = pred_mel[0].transpose(0, 1).cpu().numpy()
        target_mel = mel_spec[0].transpose(0, 1).cpu().numpy()
        # print(pred_mel), quit()
        # metric = self.criterion.mse_metric(pred_mel, mel_spec)
        data = {
            "mels": {
                "pred": pred_mel,
                "target": target_mel
            }
        }
        return data


def main(rank, world_size):
    with open("./params_test.json") as json_file:
        params = json.load(json_file)
    TesterMultiGPU.setup(rank, world_size, params)
    tester = TesterMultiGPU(params)
    tester.test()
    dist.destroy_process_group()
        

def gpu_num_checker(params):
    messege = f"Выбранное количество видеоускорителей превышает " \
            + f"максимальное количесво: ({torch.cuda.device_count()})"
    assert (params["num_gpu"] <= torch.cuda.device_count()), messege


if __name__ == "__main__":
    num_nodes = 1 
    num_gpu_per_node = torch.cuda.device_count()
    world_size = num_nodes * num_gpu_per_node
    world_size = 1
    if world_size > 1:
        mp.spawn(main, args=(world_size,), nprocs=world_size)
    else:
        TesterMultiGPU.setup_single()
        with open("./params_test.json") as json_file:
            params = json.load(json_file)
        tester = TesterMultiGPU(params)
        tester.test()



