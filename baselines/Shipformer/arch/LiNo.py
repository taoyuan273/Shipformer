import torch
import torch.nn as nn
from baselines.Shipformer.layers import FLinear, Filter, Encoder, RevIN  # 直接从 layers 目录导入
import math
import torch.nn.functional as F


class Shipformer(nn.Module):
    def __init__(self, **configs):
        super(Shipformer, self).__init__()
        self.revin_layer = RevIN(configs["enc_in"])
        self.pred_len = configs["pred_len"]
        self.Encoders = nn.ModuleList([Encoder(configs) for _ in range(configs["layers"])])
        self.embed = FLinear(configs["seq_len"]//configs["sampling_rate"], configs["d_model"])
        #self.embed = FLinear(configs["seq_len"], configs["d_model"])
        self.projection = FLinear(configs["d_model"], configs["pred_len"]//configs["sampling_rate"])
        #self.projection = FLinear(configs["d_model"], configs["pred_len"])
        self.Filter = Filter(configs["enc_in"], kernel_size=25)
        self.beta = configs["beta"]
        self.sampling_rate = configs["sampling_rate"]
        self.dropout = nn.Dropout(configs["dropout"])
        if configs["initial"]:
            self.projection.initial()

    # def forward(self, x_enc,emb=None):
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:
        #print(f"Input shape (history_data): {history_data.shape}")
        x_enc = history_data[..., 0]
        x_enc = x_enc[:, ::self.sampling_rate, :]  # 每隔 sampling_rate 个点取一个数据点
        #print(f"x_enc shape after selecting target variable: {x_enc.shape}")
        # [B, T, N]
        x_enc = self.revin_layer(x_enc, 'norm')
        #print(f"x_enc shape after revin norm: {x_enc.shape}")
        x_enc = x_enc - self.beta * self.Filter(x_enc)
        #print(f"x_enc shape after filtering: {x_enc.shape}")
        x_embed = self.embed(x_enc.transpose(1, 2))  # 在频域进行线性变换
        x_embed = self.dropout(x_embed)
        #print(f"x_embed shape after embedding: {x_embed.shape}")
        for encoder in self.Encoders:
            x_embed = encoder(x_embed)
        #print(f"x_embed shape after encoder: {x_embed.shape}")
        pred = self.projection(x_embed).transpose(1, 2)
        pred = self.dropout(pred)
        pred = self.revin_layer(pred, 'denorm')
        #print(f"pred: {pred.shape}")
        pred = F.interpolate(pred.permute(0, 2, 1), size=self.pred_len, mode='linear', align_corners=True).permute(0, 2, 1)
        
        # [B, F, N]
        # [B, F, N]

        emb = kwargs.get("emb", None)
        if emb is None:
            return pred.unsqueeze(-1)
        else:
            return pred, x_embed