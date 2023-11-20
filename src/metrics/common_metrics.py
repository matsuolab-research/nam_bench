import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# TODO: Aggregation functions should be divided into other file


def accuracy(preds: torch.Tensor, labels: torch.Tensor, metainfo: list[dict]=None):
    if preds.ndim == 2:
        preds = preds.argmax(dim=1)
    acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
    return {"acc_score": acc}


def recall(preds: torch.Tensor, labels: torch.Tensor, metainfo: list[dict]=None):
    if preds.ndim == 2:
        preds = preds.argmax(dim=1)
    recall = recall_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
    return {"recall_score": recall}


def precision(preds: torch.Tensor, labels: torch.Tensor, metainfo: list[dict]=None):
    if preds.ndim == 2:
        preds = preds.argmax(dim=1)
    precision = precision_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
    return {"precision_score": precision}


def f1(preds: torch.Tensor, labels: torch.Tensor, metainfo: list[dict]=None):
    if preds.ndim == 2:
        preds = preds.argmax(dim=1)
    f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")
    return {"f1_score": f1}


def mse(preds: torch.Tensor, labels: torch.Tensor, metainfo: list[dict]=None):
    mse_score = F.mse_loss(preds, labels).item()
    return {"mse_score": mse_score}


def mse_last_frame(preds: torch.Tensor, labels: torch.Tensor, metainfo: list[dict]=None):
    mse_score = F.mse_loss(preds[:, -1], labels[:, -1]).item() # B, S, C, H, W
    return {"mse_last_frame_score": mse_score}


def mse_last_frame_classwise(preds: torch.Tensor, labels: torch.Tensor, metainfo: list[dict]):
    mse_score = F.mse_loss(preds[:, -1], labels[:, -1], reduction="none") # B, S, C, H, W
    ndim = mse_score.ndim
    mse_score = mse_score.mean(dim=tuple(range(1, ndim)))
    
    groupwise_mse_scores = {}
    for sample_metainfo, sample_mse in zip(metainfo, mse_score):
        key = f'mse_last_frame_{sample_metainfo["target"]}'
        if key not in groupwise_mse_scores:
            groupwise_mse_scores[key] = [sample_mse]
        else:
            groupwise_mse_scores[key].append(sample_mse)
    
    for key in groupwise_mse_scores.keys():
        groupwise_mse_scores[key] = np.mean(groupwise_mse_scores[key])
    
    return groupwise_mse_scores


def mse_classwise(preds: torch.Tensor, labels: torch.Tensor, metainfo: list[dict]):
    mse_score = F.mse_loss(preds, labels, reduction="none")
    ndim = mse_score.ndim
    mse_score = mse_score.mean(dim=tuple(range(1, ndim)))
    
    groupwise_mse_scores = {}
    for sample_metainfo, sample_mse in zip(metainfo, mse_score):
        key = f'mse_{sample_metainfo["target"]}'
        if key not in groupwise_mse_scores:
            groupwise_mse_scores[key] = [sample_mse]
        else:
            groupwise_mse_scores[key].append(sample_mse)
    
    for key in groupwise_mse_scores.keys():
        groupwise_mse_scores[key] = np.mean(groupwise_mse_scores[key])
    
    return groupwise_mse_scores


def mse_metainfowise(preds: torch.Tensor, labels: torch.Tensor, metainfo: list[dict]):
    mse_score = F.mse_loss(preds, labels, reduction="none")
    ndim = mse_score.ndim
    mse_score = mse_score.mean(dim=tuple(range(1, ndim)))
    
    groupwise_mse_scores = {}
    for sample_metainfo, sample_mse in zip(metainfo, mse_score):
        col_name = "mse"
        for key, value in sample_metainfo.items():
            col_name += f'_{key}-{value}'
        if col_name not in groupwise_mse_scores:
            groupwise_mse_scores[col_name] = [sample_mse]
        else:
            groupwise_mse_scores[col_name].append(sample_mse)
    
    for col_name in groupwise_mse_scores.keys():
        groupwise_mse_scores[col_name] = np.mean(groupwise_mse_scores[col_name])
    
    return groupwise_mse_scores


def mae(preds: torch.Tensor, labels: torch.Tensor, metainfo: list[dict]=None):
    mae_score = F.l1_loss(preds, labels).item()
    return {"mae_score": mae_score}