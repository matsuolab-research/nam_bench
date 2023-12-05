import inspect
from pathlib import Path

import numpy as np
import torch
import torchvision


# TODO: このファイルはmetricsディレクトリに置くべきか微妙なので、分割を検討する
def save_imgs(
    preds: np.ndarray, labels: np.ndarray, metainfo: list[dict],
    save_dir: str, preprocess_func: callable, return_imgs: bool=False
    ):
    """Save sequence images

    Args:
        preds (np.ndarray): Predictions
        labels (np.ndarray): Ground truths
        metainfo (list[dict]): Metainfo of dataset
        save_dir (str): Directory to save images
        preprocess_func (callable): Preprocess function. This function must return a list of (img_name: str, img: PIL.Image) pairs.
        return_imgs (bool, optional): Return images or not. Defaults to False.

    Returns:
        (None | list[tuple(str, PIL.Image)]): None or list of (img_name: str, img: PIL.Image) pairs.
    """
    fn_params = inspect.signature(preprocess_func).parameters
    if len(set(fn_params.keys()) & set(("preds", "labels", "metainfo"))) != 3:
        raise ValueError(f"preprocess_func must have arguments (preds, labels, metainfo), but got {fn_params.keys()}")
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    
    img_name_img_pairs = preprocess_func(preds, labels, metainfo)
    for img_name, img in img_name_img_pairs:
        img.save(Path(save_dir)/img_name)
    
    if return_imgs:
        return img_name_img_pairs