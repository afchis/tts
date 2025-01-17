import os
import json
import argparse
from time import gmtime, strftime

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tts.dataloader.get_loader import get_loader
from tts.models.get_model import get_model
from tts.utils.get_optim import get_optim
from tts.utils.get_scheduler import get_scheduler
from tts.utils.losses import Criterions
from tts.utils.metrics import Metrics
from tts.utils.logger import Logger


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--params", type=str, default="params.json")
parser_args = parser.parse_args()


class TrainerMultiGPU:
    def __init__(self, params):
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.params = params
        self._model_init()
        self.load_model()
        self.criterions = Criterions(self.device, self.params)
        self.metrics = Metrics(self.device, self.params)
        self._get_learning_step()
        self._get_loaders()
        self.optimizer = get_optim(self, self.params)
        self.scheduler = get_scheduler(self.optimizer, self.params)
        self.logger = Logger(trainer=self)
        self.accum_iter = 0

    def _model_init(self):
        if len(self.params["target_device"]) == 0:
            self.device = "cpu"
            print(f"Initialization Trainer: CPU")
        else:
            if self.world_size == 1:
                self.device = self.params["target_device"][0]
                print(f"Initialization Trainer: GPU -> device: cuda:{self.device}")
            else:
                self.device = self.rank
                print(f"Initialization Trainer: MultiGPU -> world_size: {self.world_size}, rank: {self.rank}]")
        model = get_model(self.params).to(self.device)
        self.model_name = model.__class__.__name__
        if self.world_size > 1:
            self.model = DDP(model, device_ids=[self.device])
        else:
            self.model = model
        self.logs_path = (
            "/", "runs", os.uname().nodename,
            self.model_name, strftime(f"%d_%b_%H_%M_{self.params['json_name']}", gmtime())
        )

    def _get_loaders(self):
        self.train_loader = get_loader(self.params, stage="train")
        self.valid_loader = get_loader(self.params, stage="valid")
        # self.test_loader = get_loader(self.params, stage="test")
        self.train_len = len(self.train_loader)
        self.valid_len = len(self.valid_loader)
        self.valid_ratio = int(self.train_len / self.valid_len)

    def save_model(self):
        dir_path = (*self.logs_path, "checkpoints")
        if self.rank != 0: return None
        dir_path = os.path.join(*dir_path)
        os.makedirs(dir_path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{dir_path}/e_{self.logger.epoch:03d}.pth")

    def load_model(self):
        chk_path = os.path.join(".", "temp", "start_checkpoints")
        chk_name = self.params["checkpoint_name"]
        os.makedirs(chk_path, exist_ok=True)
        if chk_name in os.listdir(chk_path):
            model_weights = os.path.join(chk_path, chk_name)
            self.model.load_state_dict(torch.load(model_weights))
            print("Checkpoint loaded:", model_weights)

    def _get_learning_step(self):
        if self.params["network_name"] == "tacotron2":
            self.train_step = self._train_step_tacotron2
            self.valid_step = self._valid_step_tacotron2
            self.spec_iter = 0
            self.spec_save_interval = self.params["spec_save_intervals"]
            self.train_spec_path = os.path.join(*self.logs_path, "logs", "train_imgs")
            self.valid_spec_path = os.path.join(*self.logs_path, "logs", "valid_imgs")
            os.makedirs(self.train_spec_path, exist_ok=True)
            os.makedirs(self.valid_spec_path, exist_ok=True)
        elif self.params["network_name"] == "embedder":
            self.criterions.body = self.criterions.body.to(self.device)
            self.train_step = self._train_step_embedder
            self.valid_step = self._valid_step_ebmedder
        else:
            raise NotImplementedError

    @staticmethod
    def setup_single():
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        print("Setup for train on cpu or single gpu: Done.")

    @staticmethod
    def setup(rank, world_size, params):
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        # init_method = "tcp://" + params["uuid"] + "_0:1111" # for platform learning
        init_method = "tcp://localhost:11345"  # for local testing
        dist.init_process_group(backend="nccl", init_method=init_method,
                                rank=rank, world_size=world_size)
        print("Setup for Distributed Data Parallel: Done.")

    def optimizer_step(self):
        self.accum_iter += 1
        if self.params.get("accum_grad"):
            if self.accum_iter != self.params.get("accum_grad"):
                return None
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.params["policy"] == "ExponentialLR":
            if not hasattr(self, "policy_start"):
                self.policy_start = 0
            self.policy_start += 1
            if self.policy_start > 50000:
                self.scheduler.step()
        else:
            self.scheduler.step()
        self.accum_iter = 0

    def _data_to_device(self, data):
        data_to = list()
        for _data in data:
            data_to.append([item.to(self.device) for item in _data])
        return data_to

    def train(self):
        for epoch in range(1, self.params["max_epoch"]):
            self.logger.new_epoch()
            self.valid_data = iter(self.valid_loader)
            self._train_epoch(epoch)

    def _train_epoch(self, epoch):
        for iter_, data in enumerate(self.train_loader):
            data = self.train_step(data)
            self.logger.step(data, stage="train")
            if (iter_ + 1) % self.valid_ratio == 0:
                data = self.valid_step(data)
                if isinstance(data, dict):
                    self.logger.step(data=data, stage="valid")

    ########### tacotron steps ###########
    def _train_step_tacotron2(self, data):
        self.model.train()
        data = self._data_to_device(data)
        mel_spec, phoneme = data
        pred_decoder_mel, pred_post_mel, pred_stop_value = self.model(
            text_or_phoneme=phoneme, target_mels=mel_spec
        )
        loss = self.criterions.tacotron2_loss(pred_decoder_mel=pred_decoder_mel,
                                              pred_post_mel=pred_post_mel,
                                              pred_stop_value=pred_stop_value,
                                              target_mels=mel_spec,)
        mse_decoder_loss, mse_post_loss, stop_value_loss = loss
        if self.params["losses"] == "decoder": # TEMP
            loss = mse_decoder_loss + stop_value_loss # TEMP
        elif self.params["losses"] == "full": # TEMP
            loss = mse_decoder_loss + mse_post_loss + stop_value_loss # TEMP
        # if params.get("reg_lambda"):
        #     l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
        #     loss *= params["reg_lambda"] * l2_norm
        loss.backward()
        self.optimizer_step()
        if self.world_size > 1:
            dist.all_reduce(mse_decoder_loss)
            dist.all_reduce(mse_post_loss)
            dist.all_reduce(stop_value_loss)
        out = {
            "losses": {
                "decoder": mse_decoder_loss.item(),
                "post": mse_post_loss.item(),
                "stop_value": stop_value_loss.item(),
            },
            "mels": {
                "target": [_mel.transpose(0, 1).cpu() for _mel in mel_spec],
                "pred_decoder": pred_decoder_mel.transpose(1, 2).detach().cpu(),
                "pred_post": pred_post_mel.transpose(1, 2).detach().cpu(),
            }
        }
        return out

    def _valid_step_tacotron2(self, train_out):
        self.model.eval()
        self.spec_iter += 1
        if self.spec_iter % self.spec_save_interval == 0:
            self.train_loader.dataset.librosa.plot_pred(
                mels=train_out["mels"],
                epoch=self.logger.epoch,
                _iter=self.logger.iters["total"],
                output_path=self.train_spec_path,
            )
        try:
            data = next(self.valid_data)
        except StopIteration:
            return StopIteration("Valid data is end")
        with torch.no_grad():
            data = self._data_to_device(data)
            mel_spec, phoneme = data
            pred_decoder_mel, pred_post_mel, pred_stop_value = self.model(
                text_or_phoneme=phoneme, target_mels=mel_spec # TEMP: target_mels=None
            )
            loss = self.criterions.tacotron2_loss(
                pred_decoder_mel=pred_decoder_mel,
                pred_post_mel=pred_post_mel,
                pred_stop_value=pred_stop_value,
                target_mels=mel_spec,
            )
        mse_decoder_loss, mse_post_loss, stop_value_loss = loss
        if self.world_size > 1:
            dist.all_reduce(mse_decoder_loss)
            dist.all_reduce(mse_post_loss)
            dist.all_reduce(stop_value_loss)
        out = {
            "losses": {
                "decoder": mse_decoder_loss.item(),
                "post": mse_post_loss.item(),
                "stop_value": stop_value_loss.item(),
            },
            "mels": {
                "target": [_mel.transpose(0, 1).cpu() for _mel in mel_spec],
                "pred_decoder": pred_decoder_mel.transpose(1, 2).detach().cpu(),
                "pred_post": pred_post_mel.transpose(1, 2).detach().cpu(),
            }
        }
        if self.spec_iter % self.spec_save_interval == 0:
            self.valid_loader.dataset.librosa.plot_pred(
                mels=out["mels"],
                epoch=self.logger.epoch,
                _iter=self.logger.iters["total"],
                output_path=self.valid_spec_path,
            )
        return out

########### fast_speech2 steps ###########
    def _train_step_fast_speech2(self, data):
        data = self._data_to_device(data)
        mel_spec, phoneme = data
        self.optimizer.zero_grad()
        pred_mel_spec = self.model(phoneme)
        # loss = self.criterions.l1_loss(pred_mel_spec, mel_spec)
        # loss_duration = self.criterions.mse_loss(mel_lens, mel_spec)
        loss.backward()
        self.optimizer.step()

    ########### embedder steps ###########
    def _train_step_embedder(self, wavs):
        self.model.train()
        wavs = wavs.to(self.device)
        num_speakers, num_utterance, ch, seq = wavs.size()
        wavs = wavs.view(num_speakers*num_utterance, ch, seq)
        dvectors = self.model(wavs)
        dvectors = dvectors.view(num_speakers, num_utterance, dvectors.size(-1))
        loss, pos_sim, neg_sim = self.criterions(dvectors)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.criterions.body.parameters()),
            max_norm=3,
            norm_type=2.0
        )
        self.model.embedding.weight.grad *= 0.5
        self.model.embedding.bias.grad *= 0.5
        self.criterions.body.w.grad *= 0.01
        self.criterions.body.b.grad *= 0.01
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        if self.world_size > 1:
            dist.all_reduce(loss)
        out = {
            "losses": {
                "loss": loss.item(),
                "pos_sim": pos_sim,
                "neg_sim": neg_sim
            }
        }
        return out

    def _valid_step_ebmedder(self, wavs):
        self.model.eval()
        wavs = wavs.to(self.device)
        num_speakers, num_utterance, ch, seq = wavs.size()
        wavs = wavs.view(num_speakers*num_utterance, ch, seq)
        dvectors = self.model(wavs)
        dvectors = dvectors.view(num_speakers, num_utterance, dvectors.size(-1))
        loss, pos_sim, neg_sim = self.criterions(dvectors)
        eer = self.metrics(dvectors)
        if self.world_size > 1:
            dist.all_reduce(loss)
        out = {
            "losses": {
                "loss": loss.item(),
                "pos_sim": pos_sim,
                "neg_sim": neg_sim,
                "eer": eer
            }
        }
        return out

def main(rank, world_size, params):
    TrainerMultiGPU.setup(rank, world_size, params)
    trainer = TrainerMultiGPU(params)
    trainer.train()
    dist.destroy_process_group()


def gpu_num_checker(params):
    messege = f"Выбранное количество видеоускорителей превышает " \
            + f"максимальное количесво: ({torch.cuda.device_count()})"
    assert (params["num_gpu"] <= torch.cuda.device_count()), messege


if __name__ == "__main__":
    with open(os.path.join("params", parser_args.params)) as json_file:
        params = json.load(json_file)
        params["json_name"] = parser_args.params
    if len(params["target_device"]) < 2:
        TrainerMultiGPU.setup_single()
        trainer = TrainerMultiGPU(params)
        trainer.train()
    else:
        num_nodes = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in params["target_device"]])
        num_gpu_per_node = torch.cuda.device_count()
        world_size = num_nodes * num_gpu_per_node
        mp.spawn(main, args=(world_size, params,), nprocs=world_size)

