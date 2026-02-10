import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .FLinear import FLinear
from baselines.Shipformer.layers1.Transformer_EncDec import Encoder_ori, EncoderLayer
from baselines.Shipformer.layers1.SelfAttention_Family import AttentionLayer, FullAttention_ablation

class Encoder(nn.Module):
    def __init__(self, configs):
        super(Encoder, self).__init__()
        #self.fc1 = FLinear(configs["d_model"], configs["d_model"])
        self.fc2 = FLinear(configs["d_model"], configs["d_pick"])
        self.fc_core = FLinear(configs["d_pick"], configs["d_model"])
        self.fc_ori = FLinear(configs["d_model"], configs["d_model"])
        self.seq_len=configs["seq_len"]
        
        self.fc3 = FLinear(configs["d_model"], configs["d_model"])
        #self.fc4 = FLinear(configs["d_model"], configs["d_model"])
        #self.fc5 = FLinear(configs["d_model"], configs["d_model"])
        #self.fc6 = FLinear(configs["d_model"], configs["d_model"])
        self.norm1 = nn.LayerNorm(configs["d_model"])
        #self.norm2 = nn.LayerNorm(configs["d_model"])
        #self.norm3 = nn.LayerNorm(257)
        #self.norm4 = nn.LayerNorm(257)
        #self.norm5 = nn.LayerNorm(configs["d_model"])
        self.dropout = nn.Dropout(configs["dropout"])
        self.embeddings = nn.Parameter(torch.randn(1, configs["embed_size"]))
        self.embed_size = configs["embed_size"]
        self.d_model = configs["d_model"]
        self.d_pick = configs["d_pick"]
        self.valid_fre_points = int((configs["d_pick"] + 1) / 2 + 0.5)
        
        self.encoder_fre_real = Encoder_ori(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention_ablation(False, configs["factor"], attention_dropout=configs["dropout1"],
                                               output_attention=configs["output_attention"], token_num=configs["enc_in"],
                                               SF_mode=configs["attn_enhance"], softmax_flag=configs["attn_softmax_flag"],
                                               weight_plus=configs["attn_weight_plus"],
                                               outside_softmax=configs["attn_outside_softmax"],
                                               plot_mat_flag=configs["plot_mat_flag"] and _ == configs["e_layers"] - 1,
                                               plot_grad_flag=configs["plot_grad_flag"] and _ == configs["e_layers"] - 1,
                                               save_folder=f'./attn_results/{configs["plot_mat_label"]}_{configs["seq_len"]}'
                                                           f'_{configs["seq_len"]}_011_last_layer/real'),
                        configs["d_model1"], configs["n_heads"]),
                    configs["d_model1"],
                    configs["d_ff"],
                    dropout=configs["dropout1"],
                    activation=configs["activation"]
                ) for _ in range(configs["e_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(configs["d_model1"]),
            one_output=True,
            CKA_flag=configs["CKA_flag"]
        )

        self.fre_trans_real = nn.Sequential(
            nn.Linear(self.valid_fre_points * self.embed_size, configs["d_model1"]),
            nn.Dropout(configs["dropout1"]),
            self.encoder_fre_real,
            nn.Linear(configs["d_model1"], self.valid_fre_points * self.embed_size),
            nn.Dropout(configs["dropout1"])
        )

        self.encoder_fre_imag = Encoder_ori(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention_ablation(False, configs["factor"], attention_dropout=configs["dropout1"],
                                               output_attention=configs["output_attention"], token_num=configs["enc_in"],
                                               SF_mode=configs["attn_enhance"], softmax_flag=configs["attn_softmax_flag"],
                                               weight_plus=configs["attn_weight_plus"],
                                               outside_softmax=configs["attn_outside_softmax"],
                                               plot_mat_flag=configs["plot_mat_flag"] and _ == configs["e_layers"] - 1,
                                               plot_grad_flag=configs["plot_grad_flag"] and _ == configs["e_layers"] - 1,
                                               save_folder=f'./attn_results/{configs["plot_mat_label"]}_{configs["seq_len"]}'
                                                           f'_{configs["seq_len"]}_011_last_layer/imag'),
                        configs["d_model1"], configs["n_heads"]),
                    configs["d_model1"],
                    configs["d_ff"],
                    dropout=configs["dropout1"],
                    activation=configs["activation"]
                ) for _ in range(configs["e_layers"])
            ],
            norm_layer=torch.nn.LayerNorm(configs["d_model1"]),
            one_output=True,
            CKA_flag=configs["CKA_flag"]
        )

        self.fre_trans_imag = nn.Sequential(
            nn.Linear(self.valid_fre_points * self.embed_size, configs["d_model1"]),
            nn.Dropout(configs["dropout1"]),
            self.encoder_fre_imag,
            nn.Linear(configs["d_model1"], self.valid_fre_points * self.embed_size),
            nn.Dropout(configs["dropout1"])
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.d_pick * self.embed_size, configs["d_ff"]),
            nn.Dropout(configs["dropout1"]),
            nn.GELU(),
            nn.Linear(configs["d_ff"], configs["d_pick"]),
            nn.Dropout(configs["dropout1"])
        )
        
    def Fre_Trans(self, x):
        # [B, N, T, D]
        B, N, T, D = x.shape
        #assert T == self.seq_len
        assert T == self.d_pick
        # [B, N, D, T]
        x = x.transpose(-1, -2)

        # fft
        # [B, N, D, fre_points]
        x_fre = torch.fft.rfft(x, dim=-1, norm='ortho')  # FFT on L dimension
        # [B, N, D, fre_points]
        assert x_fre.shape[-1] == self.valid_fre_points

        y_real, y_imag = x_fre.real, x_fre.imag
        
        # 归一化 FFT 变换后的特征
        #y_real = self.norm3(y_real)
        #y_imag = self.norm4(y_imag)

        # ########## transformer ####
        y_real = self.fre_trans_real(y_real.flatten(-2)).reshape(B, N, D, self.valid_fre_points)
        y_imag = self.fre_trans_imag(y_imag.flatten(-2)).reshape(B, N, D, self.valid_fre_points)
        y = torch.complex(y_real, y_imag)

        # [B, N, D, T]; automatically neglect the imag part of freq 0
        x = torch.fft.irfft(y, n=T, dim=-1, norm='ortho')
        #print(f"x shape: {x.shape}")
        # 归一化 Transformer 输出
        #x = self.norm5(x)

        # [B, N, T, D]
        x = x.transpose(-1, -2)
        return x
    
    def tokenEmb(self, x, embeddings):
        #print(f"Original x shape: {x.shape}")
        if self.embed_size <= 1:
            return x.transpose(-1, -2).unsqueeze(-1)  # [B, N, T] -> [B, N, T, 1]

        # 变换维度： [B, N, T] -> [B, N, T, 1]
        x = x.transpose(-1, -2).unsqueeze(-1)
        #print(f"After transpose x shape: {x.shape}")
        

        # 扩展到 [B, N, T, D]
        return x * embeddings

    def forward(self, inp):
        #print(f"inp shape: {inp.shape}")
        batch_size, channels, d_model = inp.shape
        # set FFN
        core = F.gelu(inp)  #d_model->d_model
        core = self.fc2(core)  #d_model->d_pick->

        # stochastic pooling
        # core_fft=torch.fft.rfft(core, dim=-1)  #转到频域
        # energy = torch.abs(core_fft).pow(2)  #计算频域的能量
        # ratio = F.softmax(energy, dim=1)
        # ratio = ratio.permute(0, 2, 1)
        # ratio = ratio.reshape(-1, channels)
        # indices = torch.multinomial(ratio, 1)   #随机采样
        # indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)  #变回维度
        # core = torch.fft.irfft(torch.gather(core_fft, 1, indices),dim=-1).repeat(1, channels, 1)  #傅里叶逆变换
        
        core = core.transpose(1, 2)
        #  **维度扩展**
        core = self.tokenEmb(core, self.embeddings)
        #print(f"Shape of core: {core.shape}")


        #  **FFT + Transformer + IFFT**
        core = self.Fre_Trans(core)+core
        #print(f"Shape of core: {core.shape}")
        
        core = self.fc(core.flatten(-2)).transpose(-1, -2)
        
        core = core.transpose(1, 2)
        
        #print(f"Shape of core: {core.shape}")

        ## mlp fusion
        #core = F.gelu((self.fc_core(core)+(inp)))
        core = F.gelu((self.fc_core(core)+self.fc_ori(inp)))
        core = F.gelu(self.fc3(core+inp))
        #core = self.dropout(core)
        #print(f"Shape of core: {core.shape}")
        #core = self.fc4(core)
        res = self.norm1(inp + self.dropout(core))
        output1 = res
        output1 = self.dropout(F.gelu(output1))
        #output2 = self.dropout(self.fc6(output1))
        #output2 = self.dropout(output1)
        #print(f"output2 shape: {output2.shape}")
        
        return (res + output1)