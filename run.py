import os
import torch
import argparse

import logging
from pathlib import Path
import pandas as pd
# sys.path.append('./')
import sys
import os
# os.chdir(sys.path[0])
from scipy import spatial
import numpy as np
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from config import *
from dataset import *
from dataloads import *
from trainer import *
from evaluation import *
from DrW import *

constant = Constant(cur_time=None)
tp = 1.0
frl = 40
fll = 10

parser = argparse.ArgumentParser()
# data arguments
parser.add_argument('--train', action='store_true', help='start train')
parser.add_argument('--test', action='store_true', help='start test')
parser.add_argument('--test_single', action='store_true', help='start test single')
parser.add_argument('--save_path', type=str, default=constant.savedir)
parser.add_argument('--bin_size', type=int, default=30)
parser.add_argument('--num_neg', type=int, default=16)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--save_file', type=str, default='model.pt')
parser.add_argument('--train_percentage', type=float, default=tp)
parser.add_argument('--fix_left_length', type=int, default=fll)
parser.add_argument('--fix_right_length', type=int, default=frl)

config = parser.parse_args(['--test'])   #just for test of the paper results.

constant.logger.info(device)

# load dataset
interactions = Interactions(config, constant.logger)

# train_dataset = my_Datasets(relation=interactions.train_relation_df, interaction=interactions, stage='train', batch_size=config.train_batch_size, resample=True, bin_size=config.bin_size)
test_dataset = my_Datasets(relation=interactions.test_relation_df, interaction=interactions, stage='test', batch_size=config.test_batch_size, bin_size=config.bin_size, shuffle=False)

# trainloader = Dataloaders(
#     dataset=train_dataset,
#     stage='train',
#     fixed_length_left=config.fix_left_length,
#     fixed_length_right=config.fix_right_length,
# )
testloader = Dataloaders(
    dataset=test_dataset,
    stage='dev',
    fixed_length_left=config.fix_left_length,
    fixed_length_right=config.fix_right_length,
)

model_params = {}
model_params['mask_value'] = 0
model_params['embedding'] = interactions.embedding_matrix

model_params['top_k'] = 30
model_params['mlp_activation_func'] = 'tanh'
model_params['embedding_freeze'] = False
model_params['padding_idx'] = 0
model_params['out_activation_func'] = None

model_params['hidden_size'] = 64
model_params['num_layers'] = 1
model_params['dropout_rate'] = 0

model_params['mlp_num_layers'] = 1
model_params['mlp_num_units'] = 10
model_params['mlp_num_fan_out'] = 1


# words_idf = torch.tensor(interactions.words_idf).type(torch.float).to(device)
model = DrW(model_params=model_params)
model.build()
model.to(device)
constant.logger.info(config)
constant.logger.info(model)
str_print = 'Trainable params: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad))
constant.logger.info(str_print)
optimizer = torch.optim.Adadelta(model.parameters())

metrics = [
    NormalizedDiscountedCumulativeGain(k=3),
    NormalizedDiscountedCumulativeGain(k=5),
    NormalizedDiscountedCumulativeGain(k=10),
    MeanReciprocalRank()
]

trainer = Trainer(
    num_neg=config.num_neg,
    metrics=metrics,
    model=model,
    optimizer=optimizer,
    trainloader=None,   #just for test of the paper results.
    validloader=testloader,
    validate_interval=None,
    epochs=1,
    checkpoint=  DATA_DIR / dataname / 'model.pt',
    save_dir=config.save_path,
    constant=constant
)

if config.test:
    trainer.run_test()