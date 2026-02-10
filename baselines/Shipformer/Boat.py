import os
import sys
from easydict import EasyDict
sys.path.append(os.path.abspath(__file__ + '/../../..'))

from basicts.metrics import masked_mae, masked_mse, masked_mape, masked_rmse, masked_wape,masked_r2,composite_loss
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import Shipformer

############################## Hot Parameters ##############################
# Dataset & Metrics configuration
DATA_NAME = 'Boat'  # Dataset name
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # Length of input sequence
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # Length of output sequence
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # Train/Validation/Test split ratios
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # Whether to normalize each channel of the data
RESCALE = regular_settings['RESCALE'] # Whether to rescale the data
NULL_VAL = regular_settings['NULL_VAL'] # Null value in the data
# Model architecture and parameters
MODEL_ARCH = Shipformer
MODEL_PARAM = {
    "pred_len": OUTPUT_LEN,           # 预测序列长度
    "enc_in": 10,             # 输入通道数
    "seq_len": INPUT_LEN,            # 输入序列长度
    "embed_size": 2,                # 嵌入向量维度
    "d_model1": 128,                  # 模型隐藏层大小
    "d_ff": 256,                    # Feed-forward 层大小
    "e_layers": 1,                   # 编码器层数
    "factor": 10,                    # 默认因子
    "dropout1": 0.5,                 # Dropout
    "activation": "gelu",            # 激活函数
    "output_attention": False,       # 是否输出注意力
    "attn_enhance": 1,               # 是否增强注意力
    "attn_softmax_flag": 1,          # 是否在注意力中使用 softmax
    "attn_weight_plus": 0,           # 叠加注意力权重
    "attn_outside_softmax": 0,       # softmax 外再做处理
    "CKA_flag": 0,                   # CKA 标志
    "plot_mat_flag": 0,              # 是否绘制矩阵
    "plot_grad_flag": 0,             # 是否绘制梯度
    "temp_patch_len": 16,            # 临时 patch 长度
    "temp_stride": 8,                 # 临时步幅
    "n_heads":2,
    
    "d_pick": 128,
    #"d_pick1": 512,
    #"d_model2": 1000,
    "d_model": 512,          # Transformer 隐藏层维度
    "layers": 1,             # Transformer 编码器层数
    "dropout": 0.5,          # Dropout 比例
    "beta": 0.2,             # 低频滤波器参数
    "initial": 1,            # 是否初始化 projection 层
    "plot_mat_label": 'dataset',      # 绘图标签
    "sampling_rate": 4
}
NUM_EPOCHS = 70

############################## General Configuration ##############################
CFG = EasyDict()
# General settings
CFG.DESCRIPTION = 'An Example Config'
CFG.GPU_NUM = 1 # Number of GPUs to use (0 for CPU mode)
# Runner
CFG.RUNNER = SimpleTimeSeriesForecastingRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
# Dataset settings
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' is automatically set by the runner
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
# Scaler settings
CFG.SCALER.TYPE = ZScoreScaler # Scaler class
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
# Model settings
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################

CFG.METRICS = EasyDict()
# Metrics settings
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'MSE': masked_mse,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
    'WAPE': masked_wape,
    'R2': masked_r2,
                            })
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = composite_loss
# Optimizer settings
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0003,
    "weight_decay": 0.0001,
}
# Learning rate scheduler settings
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [1, 25,50],
    "gamma": 0.5
}
CFG.TRAIN.CLIP_GRAD_PARAM = {
    'max_norm': 5.0
}
# Train data loader settings
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64

############################## Evaluation Configuration ##############################

CFG.EVAL = EasyDict()

# Evaluation parameters
CFG.EVAL.USE_GPU = True # Whether to use GPU for evaluation. Default: True
