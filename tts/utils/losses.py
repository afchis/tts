import torch
import torch.nn as nn
import torch.nn.functional as F


class Criterions:
    def __init__(self, device, params):
        self.device = device
        self.body = self._get_criterion(params)

    def __call__(self, *args, **kwargs):
        return self.body(*args, **kwargs)

    def _get_criterion(self, params):
        if params["network_name"] == "embedder":
            return GE2ELoss(w_init=10., b_init=-5., loss=params["loss_fn"])
        elif params["network_name"] == "tacotron2":
            return self.tacotron2_loss
        else:
            raise NotImplementedError("Criterions._get_criterion")

    def tacotron2_loss(self, pred_decoder_mel, pred_post_mel, pred_stop_value, target_mels):
        with torch.no_grad():
        #     masking = torch.zeros_like(pred_post_mel)
            target_stop_values = list()
            for i, batch in enumerate(target_mels):
                # masking[i, :batch.size(0)] = 1.
                batch_stop_values = torch.zeros([batch.size(0), 1])
                target_stop_values.append(batch_stop_values)
            target_stop_values = torch.nn.utils.rnn.pad_sequence(target_stop_values,
                                                             batch_first=True,
                                                             padding_value=1.0).to(self.device)
            target_mels = torch.nn.utils.rnn.pad_sequence(target_mels, batch_first=True)
        mse_decoder_loss = F.mse_loss(pred_decoder_mel, target_mels)
        mse_post_loss = F.mse_loss(pred_post_mel, target_mels)
        stop_value_loss = F.binary_cross_entropy(pred_stop_value, target_stop_values)
        return mse_decoder_loss, mse_post_loss, stop_value_loss

    def mse_metric(self, pred_mel, target_mel):
        out = torch.nn.utils.rnn.pad_sequence([pred_mel] + target_mel, batch_first=True)
        pred_mel, target_mel = out[0], out[1]
        return F.mse_loss(pred_mel, target_mel)

    # def duration_loss(self, mel_lens, target_mels):
    #     target_lens = list()
    #     for mel in target_mels:
    #         target_lens.append(mel.size(-1))
    #     print(mel_lens, target_lens)
    #     i = 0
    #     loss = 0
    #     for target, pred in zip(target_lens, mel_lens):
    #         loss += (target - pred)**2
    #         i += 1
    #     loss = loss / i 
    #     return loss

    # def l1_loss(self, pred, target):
    #     loss = list()
    #     for _pred, _target in zip(pred, target):
    #         # _pred = F.pad(_pred, (0, _target.size(-1)-_pred.size(-1), 0, 0), "constant", 0)
    #         # loss.append(F.l1_loss(_pred, _target, reduction="mean"))
    #         print(_pred.shape, _target.shape)
    #         loss.append(F.l1_loss(_pred, _target[:, :_pred.size(1)], reduction="mean"))
    #     return torch.stack(loss)


class GE2ELoss(nn.Module):
    def __init__(self, device="cpu", w_init=10., b_init=-5., loss="softmax"):
        super().__init__()
        self.w = nn.Parameter(torch.FloatTensor([w_init]))
        self.b = nn.Parameter(torch.FloatTensor([b_init]))
        self.w.requires_grad = True
        self.b.requires_grad = True
        if loss == "softmax":
            self.embed_loss = self._softmax_loss
        elif loss == "contrans":
            self.embed_loss = self._contrast_loss
        else:
            raise Exception("Wrong loss mode for GE2ELoss, must be --> ['softmax', 'contrast']")

    def _softmax_loss(self, cos_sim_matrix):
        losses = list()
        for sp in range(cos_sim_matrix.size(0)):
            for ut in range(cos_sim_matrix.size(1)):
                loss = - cos_sim_matrix[sp, ut, sp] + torch.log(torch.exp(cos_sim_matrix[sp, ut]).sum())
                losses.append(loss)
        return torch.stack(losses)

    def _contrast_loss(self, ):
        raise NotImplementedError

    def _cosine_similarity(self, dvectors):
        n_spkr, n_uttr, dim = dvectors.size()
        centers = dvectors.mean(dim=1).reshape(n_spkr, 1, 1, dim).expand(n_spkr, n_uttr, n_spkr, dim)
        centers_uttr = dvectors.sum(dim=1)
        centers_uttr = centers_uttr.unsqueeze(1).expand(n_spkr, n_uttr, dim)
        centers_uttr = (centers_uttr - dvectors) / (n_uttr - 1)
        for sp in range(n_spkr):
            for ut in range(n_uttr):
                centers[sp, ut, sp] = centers_uttr[sp, ut]
        dvectors = dvectors.unsqueeze(2).expand(n_spkr, n_uttr, n_spkr, dim)
        output = torch.cosine_similarity(dvectors, centers, dim=-1, eps=1e-6)
        return output

    def _neg_centroids(self, dvec):
        return torch.mean(dvec, dim=1, keepdim=True)

    def _pos_centroid(self, dvec, sp_idx, utt_idx):
        pos_cent = torch.cat([dvec[sp_idx, :utt_idx], dvec[sp_idx, utt_idx+1:]], dim=0)
        return torch.mean(pos_cent, dim=0, keepdim=True)

    def _sim_matrix(self, dvec):
        neg_centroids = self._neg_centroids(dvec)
        '''
        dvec.shape          --> [speaker_idx, utterance_idx, emb_dim]
        neg_cintroids.shape --> [speaker_idx, 1, emb_dim]
        pos_centroid.shape  --> [1, emb_dim]
        '''
        pos_sim, neg_sim = list(), list()
        pos_mean, neg_mean = 0, 0
        for sp_idx in range(dvec.size(0)):
            pos_sim_speaker = list()
            neg_sim_speaker = list()
            for utt_idx in range(dvec.size(1)):
                # pos sim:
                pos_centroid = self._pos_centroid(dvec, sp_idx, utt_idx)
                sim = F.cosine_similarity(dvec[sp_idx, utt_idx].reshape(1, -1), pos_centroid, dim=1, eps=1e-6)
                pos_mean += sim.mean().item()
                pos_sim_utt = self.w * sim + self.b # [1]
                pos_sim_speaker.append(pos_sim_utt)
                # neg sim:
                sim = F.cosine_similarity(dvec[sp_idx, utt_idx].reshape(1, -1), torch.cat([neg_centroids[:sp_idx], neg_centroids[sp_idx+1:]], dim=0).squeeze(), dim=1)
                neg_mean += sim.mean().item()
                neg_sim_utt = self.w * sim  + self.b # [speaker_idx-1]
                neg_sim_speaker.append(neg_sim_utt)
            pos_sim_speaker = torch.stack(pos_sim_speaker, dim=0)
            pos_sim.append(pos_sim_speaker)
            neg_sim_speaker = torch.stack(neg_sim_speaker, dim=0)
            neg_sim.append(neg_sim_speaker)
        pos_sim = torch.stack(pos_sim, dim=0) # [speaker_idx, utterance_idx, 1]
        neg_sim = torch.stack(neg_sim, dim=0) # [speaker_idx, utterance_idx, speaker_idx-1]
        pos_mean /= dvec.size(0) * dvec.size(1)
        neg_mean /= dvec.size(0) * dvec.size(1)
        return pos_sim, neg_sim, pos_mean, neg_mean

    def _softmax_loss(self, pos_sim, neg_sim):
        loss = - pos_sim.squeeze() + torch.log(torch.exp(neg_sim).sum(dim=2))
        return loss.mean() #, torch.log(torch.exp(neg_sim).sum(dim=2)).mean().item()

    def forward(self, dvectors):
        # dvectors shape --> [num_speaker, num_utterance, feat_dim]
        pos_sim, neg_sim, pos_mean, neg_mean = self._sim_matrix(dvectors)
        torch.clamp(self.w, 1e-6)
        loss = self._softmax_loss(pos_sim, neg_sim)
        return loss, pos_mean, neg_mean
        # cos_sim_matrix = self._cosine_similarity(dvectors)
        # cos_sim_matrix = cos_sim_matrix * self.w + self.b
        # loss = self.embed_loss(cos_sim_matrix)
        # return loss.sum(), 0, 0


if __name__ == "__main__":
    dvectors = torch.rand([64, 10, 256])
    criterion = GE2ELoss()
    loss = criterion(dvectors)
    print(loss)
