import time
import torch

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


class Metrics:
    def __init__(self, device, params):
        self.device = device
        self.body = self._get_metrics(params)

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        return self.body(*args, **kwargs)

    def _get_metrics(self, params):
        if params["network_name"] == "embedder":
            return self.EqualErrorRate
        elif params["network_name"] == "tacotron2":
            return None
        else:
            raise NotImplementedError("Metrics._get_metrics")

    def _get_cos_sim_scores(self, dvectors):
        dvectors = dvectors.detach()
        num_speakers, num_utterance, dim = dvectors.size()
        dvectors = dvectors.tolist()
        dvectors = [[torch.tensor(utt) for utt in sp] for sp in dvectors]
        positive_scores, negative_scores = list(), list()
        for s, speaker in reversed(list(enumerate(dvectors))):
            for u, utterance in reversed(list(enumerate(speaker))):
                for s_, speaker_ in enumerate(dvectors):
                    for u_, utterance_ in enumerate(speaker_):
                        score = torch.cosine_similarity(utterance.unsqueeze(0),
                                                        utterance_.unsqueeze(0))
                        if s == s_:
                            if u == u_:
                                pass
                            else:
                                positive_scores.append(score)
                        else:
                            negative_scores.append(score)
                dvectors[s].pop(u)
        positive_scores = torch.cat(positive_scores)
        negative_scores = torch.cat(negative_scores)
        return positive_scores, negative_scores

    def _get_cos_sim_scores_optim(self, dvectors):
        num_speakers, num_utterance, dim = dvectors.size()
        utterances = dvectors.reshape(num_speakers * num_utterance, dim)
        cos_sim = torch.mm(utterances, utterances.t())
        speaker_idx = torch.arange(num_speakers).repeat_interleave(num_utterance)
        utterance_idx = torch.arange(num_utterance).repeat(num_speakers)
        positive_scores = cos_sim[speaker_idx, utterance_idx]
        negative_scores = cos_sim[speaker_idx != speaker_idx.view(-1, 1)].view(-1)
        return positive_scores, negative_scores

    def EqualErrorRate(self, dvectors):
        # t = time.time()
        pos_score, neg_score = self._get_cos_sim_scores(dvectors)
        # t1 = time.time() - t
        all_scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        fpr, tpr, thresholds = roc_curve(labels.cpu(), all_scores.cpu())
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        # print("defalut: ", eer, t1)
        # t = time.time()
        # pos_score, neg_score = self._get_cos_sim_scores_optim(dvectors)
        # t2 = time.time() - t
        # all_scores = torch.cat([pos_score, neg_score])
        # labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        # fpr, tpr, thresholds = roc_curve(labels.cpu(), all_scores.cpu())
        # eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        # print("optim: ", eer, t2)
        return eer
    

if __name__ == "__main__":
    params = {"network_name": "embedder"}
    metrics = Metrics(device="cpu", params=params)
    dvectors = torch.rand([64, 10, 256])
    err = metrics(dvectors)
    # print(err)


