import os
import os.path as osp
import json
from datetime import datetime
import warnings
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nam_bench import utils


class Config:
    """Config class for evaluation"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        tmp_config = self._load_config(config_path)
        self._sanitiy_check(tmp_config)
        for property, value in vars(tmp_config).items():
            setattr(self, property, value)
    
    def _load_config(self, config_path: str) -> object:
        """Load config from json file

        Args:
            config_path (str): A path to json file which contains config for evaluation.

        Raises:
            ValueError: Not a path to json file

        Returns:
            object: A object of config
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(
                    f,
                    object_hook=lambda x: SimpleNamespace(**x),
                )
        except:
            raise ValueError("config_path must be a path to json file but got {}".format(config_path))
        
        return config
        
    def _sanitiy_check(self, config: object):
        if not hasattr(config, "batch_size"):
            warnings.warn("batch_size is not in config file. Set batch_size to 32")
            self.batch_size = 32

        if not hasattr(config, "dataset_config"):
            raise ValueError("dataset_config is not in config file. Please specify dataset_config in config file")
            
        if not hasattr(config.dataset_config, "dataset_name"):
            raise ValueError("dataset_name is not in config file. Please specify dataset_config.name in config file")
        
        if not hasattr(config, "metrics"):
            raise ValueError("metrics is not in config file. Please specify metrics in config file")


class Evaluation:
    def __init__(self, model: callable, config_path: str):
        """Evaluation class for evaluating model performance

        Args:
            model (callable): A callable model which takes input of torch.Tensor and returns torch.Tensor
            config_path (str): A path to json file which contains config for evaluation.
        """
        self.model = model
        self.config = Config(config_path)

        self.dataset = utils.get_dataset(self.config.dataset_config) # {"dataset": Dataset, "name": str}
        self.evaluator = utils.Evaluator(utils.get_metrics(self.config.metrics)) # callable which takes input of tuple of torch.Tensors and returns list[dict]
        self.test_loader = DataLoader(
            self.dataset.data,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        self.callbacks = None
        if hasattr(self.config, "callbacks"):
            self.callbacks = utils.get_callbacks(self.config.callbacks)


    def evaluate(self, output_dir: str) -> None:
        """Evaluate model performance and dump result to csv or json file

        Args:
            output_dir (str): Output directory path
        """
        now_str = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = osp.join(output_dir, now_str)
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        print(f"Evaluation result will be saved to {output_dir}")
        
        preds = []
        ground_truths = []
        for X, y in self.test_loader:
            pred = self.model(X)
            preds.append(pred)
            ground_truths.append(y)
        
        preds = torch.cat(preds, dim=0)
        ground_truths = torch.cat(ground_truths, dim=0)
        
        if self.callbacks is not None:
            callbacks_output_dir = osp.join(output_dir, "imgs")
            if not osp.exists(callbacks_output_dir):
                os.makedirs(callbacks_output_dir)
            for callback in self.callbacks:
                callback(
                    preds=preds,
                    inputs=ground_truths,
                    metainfo=self.dataset.metainfo,
                    output_dir=callbacks_output_dir,
                )

        reports = self.evaluator(preds, ground_truths, self.dataset.metainfo)
        self._dump_result(output_dir, reports, save_as="csv")
    
    def _dump_result(
            self,
            output_dir: str,
            reports: list,
            save_as: str = "csv",
            verbose: bool = True,
        ) -> None:
        """Dump result to csv or json file (for now)

        Args:
            output_dir (str): Output directory path
            preds (list): list of predictions list[torch.Tensor]
            reports (list): list of reports list[dict]
            save_as (str, optional): Format of save file. Defaults to "csv".
            verbose (bool, optional): Defaults to True.

        Raises:
            ValueError: _description_
        """
        
        if save_as not in ["csv", "json"]:
            raise ValueError("save_as must be csv or json but got {}".format(save_as))
        
        if save_as == "csv":
            save_path = os.path.join(output_dir, f"result.csv")
            for key in reports.keys():
                if not isinstance(reports[key], list):
                    reports[key] = [reports[key]]
            df = pd.DataFrame.from_dict(reports, orient="columns")
            df.sort_index(axis=1, inplace=True)
            df.to_csv(save_path, index=False)
            
            if verbose:
                print("Dumped result to {}".format(save_path))
            return
        
        if save_as == "json":
            save_path = os.path.join(output_dir, f"result.json")
            with open(save_path, "w") as f:
                json.dump({
                    "reports": reports
                }, f)
            if verbose:
                print("Dumped result to {}".format(save_path))
            return
        