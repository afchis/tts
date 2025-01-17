import os
import time
import shutil


class Logger:
    def __init__(self, trainer):
        curr_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.gmtime())
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.lr = trainer.params["learning_rate"]
        self.num_epoch = trainer.params["max_epoch"]
        self.epoch_iters = (len(trainer.train_loader) + len(trainer.valid_loader))
        self.total_iters = trainer.params["max_epoch"] * self.epoch_iters
        message =  f"Learning starting. Time: {curr_time}. "
        message += f"Num epochs: {trainer.params['max_epoch']}. "
        message += f"Start LR: {self.lr}"
        self.init()

    def init(self):
        self.progress = 0.0
        self.epoch = 0
        self.iters = {
            "total": 0,
            "epoch": 0,
            "train": 0,
            "valid": 0,
        }
        self.losses = {
            "train": dict(),
            "valid": dict(),
        }

    def new_epoch(self):
        self.epoch += 1
        self.iters["epoch"] = 0
        self.iters["train"] = 0
        self.iters["valid"] = 0
        self.losses["train"] = dict()
        self.losses["valid"] = dict()

    def _print_message(self, message, last_iter):
        terminal_width = shutil.get_terminal_size().columns
        if len(message) > terminal_width:
            message = message[:terminal_width]
        print(" " * terminal_width, end="\r")
        print(message, end = "" if last_iter else "\r")

    def _get_losses(self, losses_dict, last_iter=False):
        loss_message = str()
        if len(losses_dict) == 0:
            loss_message = "None"
        else:
            for loss_name, loss in losses_dict.items():
                if last_iter:
                    l = sum(loss) / len(loss) / self.world_size
                    losses_dict[loss_name] = [losses_dict[loss_name][-1]]
                else:
                    l = loss[-1] / self.world_size
                loss_message += f"{loss_name}: {l:.4} "
        return loss_message[:-1]

    def _iter_message(self, last_iter=False):
        train_losses = self._get_losses(self.losses["train"], last_iter)
        valid_losses = self._get_losses(self.losses["valid"], last_iter)
        if last_iter:
            message  = f"Epoch: {self.epoch}, "
            message += f"Losses: train: [{train_losses}] || valid: [{valid_losses}]"
        else:
            message  = f"Prg: [{self.progress:.3f}{chr(37)}] "
            message += f"It: {self.iters['total']}: "
            message += f"L -> trn: [{train_losses}] || vld: [{valid_losses}]"
        if self.rank == 0: self._print_message(message, last_iter)

    def step(self, data, stage="train"):
        self.iters[stage] += 1
        self.iters["epoch"] += 1
        self.iters["total"] += 1
        self.progress = self.iters["total"] / self.total_iters * 100
        for loss_name, loss in data["losses"].items():
            if self.losses[stage].get(loss_name) is None:
                self.losses[stage][loss_name] = list()
            self.losses[stage][loss_name].append(loss)
        self._iter_message(last_iter=self.iters["epoch"]==self.epoch_iters)

    def graph_write(self, train_loss, valid_loss):
        raise NotImplementedError


class TestLogger:
    def __init__(self, tester):
        pass
