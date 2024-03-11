import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import torch
import torch.nn.functional as F
import os.path as osp
import random


def train_test_split(ann_file, ratio=0.8, seed=61):
    random.seed(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin:
            line_split = line.strip().split(",")
            filename, _, _, label = line_split
            label = float(label)
            filename = osp.join(filename)
            video_infos.append(dict(filename=filename, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def performance_fit(y_label, y_output):
    y_output_logistic = fit_function(y_label, y_output)
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic-y_label) ** 2).mean())

    return PLCC, SRCC, KRCC, RMSE


def performance_no_fit(y_label, y_output):
    PLCC = stats.pearsonr(y_output, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_label-y_label) ** 2).mean())

    return PLCC, SRCC, KRCC, RMSE



class L1RankLoss_nobatch(torch.nn.Module):
    """
    L1 loss + Rank loss
    """

    def __init__(self, **kwargs):
        super(L1RankLoss_nobatch, self).__init__()
        self.l1_w = kwargs.get("l1_w", 1)
        self.rank_w = kwargs.get("rank_w", 1)
        self.hard_thred = kwargs.get("hard_thred", 1)
        self.use_margin = kwargs.get("use_margin", False)

    def forward(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        # l1 loss
        l1_loss = F.l1_loss(preds, gts) * self.l1_w

        # simple rank
        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
            rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds - preds_t))
        else:
            rank_loss = masks_hard * torch.relu(- masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)
        loss_total = l1_loss + rank_loss * self.rank_w
        return loss_total

class MSELoss(torch.nn.Module):
    """
    Mean Squared Error (MSE) loss
    """

    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()
        self.mse_w = kwargs.get("mse_w", 1)

    def forward(self, preds, gts):
        mse_loss = F.mse_loss(preds, gts) * self.mse_w
        return mse_loss

class L1RankLoss(torch.nn.Module):


    def __init__(self, **kwargs):
        super(L1RankLoss, self).__init__()
        self.l1_w = kwargs.get("l1_w", 1)
        self.rank_w = kwargs.get("rank_w", 1)
        self.mse_w = kwargs.get("mse_w", 1)
        self.hard_thred = kwargs.get("hard_thred", 1)
        self.use_margin = kwargs.get("use_margin", False)
        self.batchsize = kwargs.get("batchsize", 8);

    def forward(self, preds, gts):
        preds = torch.reshape(preds, [self.batchsize, -1])
        preds = torch.mean(preds, dim=1)
        gts = gts.view(-1)
        # l1 loss
        l1_loss = F.l1_loss(preds, gts) * self.l1_w

        # simple rank
        n = len(preds)
        preds = preds.unsqueeze(0).repeat(n, 1)
        preds_t = preds.t()
        img_label = gts.unsqueeze(0).repeat(n, 1)
        img_label_t = img_label.t()
        masks = torch.sign(img_label - img_label_t)
        masks_hard = (torch.abs(img_label - img_label_t) < self.hard_thred) & (torch.abs(img_label - img_label_t) > 0)
        if self.use_margin:
            rank_loss = masks_hard * torch.relu(torch.abs(img_label - img_label_t) - masks * (preds - preds_t))
        else:
            rank_loss = masks_hard * torch.relu(- masks * (preds - preds_t))
        rank_loss = rank_loss.sum() / (masks_hard.sum() + 1e-08)

        loss_total = l1_loss + rank_loss * self.rank_w

        return loss_total