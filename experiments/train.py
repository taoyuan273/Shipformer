# Run a baseline model in BasicTS framework.
# pylint: disable=wrong-import-position
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.abspath(__file__ + '/../..'))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

import basicts

torch.set_num_threads(4) # aviod high cpu avg usage

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model')
    parser.add_argument('-c', '--cfg', default='baselines/Shipformer/Boat.py', help='training config')
    parser.add_argument('-g', '--gpus', default='0', help='visible gpus')
    return parser.parse_args()

def main():
    args = parse_args()
    #print("MODEL PARAM:", args.cfg['MODEL']['PARAM'])
    basicts.launch_training(args.cfg, args.gpus, node_rank=0)


if __name__ == '__main__':
    main()

    
#python experiments/train.py -c baselines/Shipformer/Boat.py --gpus '0'