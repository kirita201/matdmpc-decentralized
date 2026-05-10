# algorithm/helper.py
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

__REDUCE__ = lambda b: 'mean' if b else 'none'

def l1(pred, target, reduce=False):
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))

def mse(pred, target, reduce=False):
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))

def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def ema(m, m_target, tau):
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)

def set_requires_grad(net, value):
    for param in net.parameters():
        param.requires_grad_(value)

class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]), act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]), act_fn,
        nn.Linear(mlp_dim[1], out_dim))

def linear_schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)

def get_comm_graph(obs, comm_range, pos_slice=(2, 4)):
    """
    観測テンソルからエージェント間の通信可能性を示す隣接マスクを生成する。
 
    各エージェントの絶対位置は obs の pos_slice インデックス範囲から取得する。
    MPE の観測フォーマットは [vel(2), pos(2), ...] なので、デフォルト pos_slice=(2,4)。
 
    Parameters
    ----------
    obs : Tensor, shape (..., N, obs_dim)
        先頭次元は任意（B, B*N など）。N はエージェント数。
    comm_range : float
        通信可能距離の上限。float('inf') を渡せば全結合（ベースライン再現）。
    pos_slice : tuple of (int, int)
        obs の最終次元から位置座標を切り出すスライス範囲。
 
    Returns
    -------
    adj_mask : BoolTensor, shape (..., N, N)
        adj_mask[..., i, j] = True  ⟺  エージェント i から j へ通信可能
        対角要素（自己）は常に True。
    """
    # 位置座標を抽出: (..., N, 2)
    pos = obs[..., pos_slice[0]:pos_slice[1]]
 
    # ペアワイズ距離: (..., N, N)
    # pos_i - pos_j を計算するため次元を展開してブロードキャスト
    diff = pos.unsqueeze(-2) - pos.unsqueeze(-3)   # (..., N, N, 2)
    dist = torch.norm(diff, dim=-1)                # (..., N, N)
 
    # 通信範囲以内 or 自己 → True
    adj_mask = dist <= comm_range                  # (..., N, N)  対角は dist=0 なので常にTrue
 
    return adj_mask