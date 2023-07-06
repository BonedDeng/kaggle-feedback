import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.nn import functional as F
from sklearn.model_selection import KFold
import tqdm
warnings.filterwarnings("ignore")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(2021)


if __name__ == "__main__":
    NUM_JOBS = 12

    df = pd.read_csv("./feedback-prize-2021/train.csv").reset_index(drop=True)

    # len_df = len(df)
    # B_index = np.random.choice(len(df), int(len(df) * 0.1), replace=True)
    # df['kfold'] = int(0)
    # df.loc[B_index, 'kfold'] = int(1)
    # print(len_df, len(B_index))

    folds = KFold(n_splits=14, shuffle=True, random_state=2021).split(range(len(df)), range(len(df)))  # 多折
    for fold, (trn_idx, val_idx) in enumerate(folds):
        print(fold)
        df.loc[val_idx, 'kfold'] = int(fold)


    print(len(df))
    train_df = df[df['kfold'] == int(0)].reset_index(drop=True)
    val_df = df[df['kfold'] != int(0)].reset_index(drop=True)
    print(len(train_df), len(val_df))
    df.to_csv('./feedback-prize-2021/train_fold.csv', index=None)


    train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)

