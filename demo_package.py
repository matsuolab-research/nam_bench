from pprint import pprint

import numpy as np
import torch
import torchvision

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
        img = np.concatenate([preds_i, labels_i], axis=1)
        img = torch.from_numpy(img).float()
        img = torchvision.transforms.functional.to_pil_image(img)
        img_name = f"idx-{i}_target-{metainfo_i['target']}.png"
        yield img_name, img


if __name__ == "__main__":
    eval_op = nam_bench.Evaluator()
    
    eval_op.set_dataset_fn("MovingDigits")
    x = eval_op.get_eval_dataset(
        num_train_data=10, # NOTE: This parameter does not influence the result of evaluation.
        num_test_data=10*10,
        img_size=20,
        seq_length=4,
        obj_scales = [1.0, 1.25, 1.5, 1.25], 
        # obj_scales = [1.0, 1.25, 1.5, 1.25], 
        # obj_scales=[1.0, 1.0, 1.0, 1.0], 
        obj_rotation_speeds=[0], 
        # obj_rotation_speeds=[0, 90], 
        random_start=False
    )
    
    # Add custom metrics here.
    eval_op.add_custom_eval_fn("mean", sample_custom_eval)
    save_imgs_kwargs = {"save_dir": "./imgs", "preprocess_func": img_preprocess_fn}
    eval_op.add_custom_eval_fn("save_imgs", nam_bench.metrics.make_imgs.save_imgs, **save_imgs_kwargs)
    
    # NOTE: You can setup evaluator with config file as well.
    # config = nam_bench.eval_utils.Config(config_path=config_path) 
    # eval_op.setup_with_config(config)
    
    print(eval_op) # CHeck the evaluation configuration.
    
    # NOTE: You can preprocess for given data here.
    x = torch.from_numpy(x).float()
    # NOTE: Implement your model here and get predictions.
    model = lambda x: torch.clamp(x[:, -2] + 0.1*torch.randn_like(x[:, -1]), 0, 1) # Reconstruction model
    # model = lambda x: torch.randint(0, 10, size=(len(x), )) # Classification model
    
    preds = model(x)
    preds = preds.detach().numpy()

    reports = eval_op.evaluate(preds)
    # pprint(reports) # This is the evaluation result. You can use it for loggers, etc.
    # NOTE: Dataset metainfo can be long, so we do not print it here.
    pprint({key: value for key, value in reports.items() if key != "metainfo"})
    # You can use above reports for loggers, etc.
