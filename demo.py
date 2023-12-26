from pprint import pprint
import io

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image

import nam_bench


def sample_custom_eval(preds: np.ndarray, labels: np.ndarray, metainfo: list[dict]=None):
    """Sample custom evaluation function. Here we implement mean function.

    Args:
        preds (np.ndarray): Predictions
        labels (np.ndarray): Ground truths
        metainfo (list[dict], optional): Metainfo. Defaults to None.

    Returns:
        dict: Evaluation results
    """
    return np.mean(preds)


def img_preprocess_fn(preds: np.ndarray, labels: np.ndarray, metainfo: list[dict]=None):
    if metainfo is None:
        metainfo = [{} for _ in range(len(preds))]
    for i, (preds_i, labels_i, metainfo_i) in enumerate(zip(preds, labels, metainfo)):
        # 1, H, W
        # img = np.concatenate([preds_i, labels_i], axis=1)
        # img = torch.from_numpy(img).float()
        # img = torchvision.transforms.functional.to_pil_image(img)
        
        # Use matplotlib
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        img_pred = ax[0].imshow(preds_i, origin="lower")
        # fig.colorbar(img_pred, ax=ax[0])
        ax[0].set_title("Prediction")
        img_gt = ax[1].imshow(labels_i, origin="lower")
        # fig.colorbar(img_gt, ax=ax[1])
        ax[1].set_title("Ground Truth")
        # Transform to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img_name = f"idx-{i}.png"
        yield img_name, img


if __name__ == "__main__":
    # NOTE: Modify config object if you want to change config, but not recommended for reproducibility.
    # config = eval_utils.Config(config_path=config_path) 
    
    eval_op = nam_bench.Evaluator()
    
    # NOTE: if you want to use both train and test data, use get_dataset() instead of get_eval_dataset().
    
    # MovingDigits Example
    ###########################################################################
    # eval_op.set_dataset_fn("MovingDigits")
    # x = eval_op.get_eval_dataset(
    #     num_train_data=10, # NOTE: This parameter does not influence the result of evaluation.
    #     num_test_data=10*10,
    #     img_size=20,
    #     seq_length=4,
    #     obj_scales = [1.0, 1.25, 1.5, 1.25], 
    #     # obj_scales = [1.0, 1.25, 1.5, 1.25], 
    #     # obj_scales=[1.0, 1.0, 1.0, 1.0], 
    #     obj_rotation_speeds=[0], 
    #     # obj_rotation_speeds=[0, 90], 
    #     random_start=False
    # )
    ###########################################################################
    
    # MovingBox Example
    ###########################################################################
    eval_op.set_dataset_fn("MovingBox")
    fix_obj_widths = np.array((15, 10, 5))
    fix_obj_heights = np.full_like(fix_obj_widths, 5)
    x = eval_op.get_eval_dataset(
        num_train_data=20,
        num_test_data=10,
        random_objs=False,
        seq_len=20, 
        num_objs=3,
        frame_size=(20, 20),
        fix_obj_heights=fix_obj_heights, 
        fix_obj_widths=fix_obj_widths, 
        image=True,
        normalize=False, # 0 ~ 255 or 0 ~ 1
    )
    x = x.astype(np.float32) # Default dtype is np.float32 [0, 1]
    ###########################################################################
    
    # Add custom metrics here.
    eval_op.add_custom_eval_fn("mean", sample_custom_eval)
    save_imgs_kwargs = {"save_dir": "./imgs", "preprocess_func": img_preprocess_fn}
    eval_op.add_custom_eval_fn("save_imgs", nam_bench.metrics.make_imgs.save_imgs, **save_imgs_kwargs)
    
    print(eval_op) # CHeck the evaluation configuration.
    
    # NOTE: You can preprocess for given data here.
    x = torch.from_numpy(x).float()
    # NOTE: Implement your model here and get predictions.
    model = lambda x: torch.clamp(x[:, -2] + 0.1*torch.randn_like(x[:, -1]), 0, 1) # Reconstruction model
    # model = lambda x: torch.randint(0, 10, size=(len(x), )) # Classification model
    
    preds = model(x)
    preds = preds.detach().numpy()

    output_dict = eval_op.evaluate(preds)
    # pprint(output_dict) # This is the evaluation result. You can use it for loggers, etc.
    # NOTE: Dataset metainfo can be long, so we do not print it here.
    pprint({key: value for key, value in output_dict.items() if key != "metainfo"})
