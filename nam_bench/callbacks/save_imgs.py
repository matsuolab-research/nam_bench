import os
import os.path as osp

import torch
import torchvision.utils as vutils


def save_imgs(
    output_dir: str, 
    filename: str,
    imgs: torch.Tensor,
):
    save_path = osp.join(output_dir, filename)
    vutils.save_image(imgs, save_path)


def save_input_imgs(
    inputs: torch.Tensor,
    preds: torch.Tensor,
    metainfo: list[dict],
    output_dir: str,
):
    for img_id, (each_img, each_metainfo) in enumerate(zip(inputs, metainfo)):
        file_name = ""
        for key, value in each_metainfo.items():
            file_name += f"{key}-{value}_"
        file_name += f"input_{img_id}.png"
        
        save_imgs(output_dir, file_name, each_img)


def save_output_imgs(
    inputs: torch.Tensor,
    preds: torch.Tensor,
    metainfo: list[dict],
    output_dir: str,
):
    for img_id, (each_img, each_metainfo) in enumerate(zip(preds, metainfo)):
        file_name = ""
        for key, value in each_metainfo.items():
            file_name += f"{key}-{value}_"
        file_name += f"pred_{img_id}.png"
        
        save_imgs(output_dir, file_name, each_img)


def save_input_output_imgs(
    inputs: torch.Tensor,
    preds: torch.Tensor,
    metainfo: list[dict],
    output_dir: str,
):
    for img_id, (input_imgs, output_imgs, each_metainfo) in enumerate(zip(inputs, preds, metainfo)):
        file_name = ""
        for key, value in each_metainfo.items():
            file_name += f"{key}-{value}_"
        file_name += f"input_output_{img_id}.png"

        save_imgs(output_dir, file_name, torch.cat([input_imgs, output_imgs], dim=-2))