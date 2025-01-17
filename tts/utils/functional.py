import typing as t

import torch.jit
import torch.nn.functional as F


@torch.jit.script
def _pad_right(x: t.List[torch.Tensor], value: float, tmax: int):
    # max_len = torch.jit.annotate(t.List[int], [])
    # for i in x:
        # max_len.append(i.size(0))
    # max_len = max(max_len)
    out = list()
    for batch in x:
        batch = F.pad(batch, pad=(0, 0, 0, tmax-batch.shape[0]), value=value)
        out.append(batch)
    return torch.stack(out)

