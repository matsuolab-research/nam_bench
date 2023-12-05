import os
import os.path as osp
import json
from datetime import datetime
import warnings
from types import SimpleNamespace
from collections import OrderedDict
import pprint
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from . import utils, const

# TODO: metricsという名前ではあるが、画像化の処理なども含まれているので、名前を変更/分割する


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
        if not hasattr(config, "dataset_config"):
            raise ValueError("dataset_config is not in config file. Please specify dataset_config in config file")
            
        if not hasattr(config.dataset_config, "dataset_name"):
            raise ValueError("dataset_name is not in config file. Please specify dataset_config.name in config file")


class Evaluator:
    def __init__(self, config: Config|None = None):
        """Evaluator class

        Args:
            config (Config | None, optional): Evaluator configuration. Defaults to None.
        """
        self.config = config
        
        # TODO: 以下のdatasetに関する処理は、Datasetクラスを作成して、そこに移す
        self.dataset_name = None
        self._dataset_fn = lambda x: x
        self.__dataset_fn_kwargs = {}
        self.__dataset_metadata = None
        self.__target = None
        self._default_eval_functions = {}
        self._custom_eval_functions = {}
    
    def setup_with_config(self):
        """Setup evaluator with config
        """
        self.set_dataset_fn(self.config.dataset_config.dataset_name)
        self.__dataset_fn_kwargs = self.config.dataset_config.kwargs
    
    def get_config(self):
        """Get config

        Returns:
            Config: A config
        """
        return {
            "dataset_name": self.dataset_name,
            "dataset_fn": self._dataset_fn.__name__,
            "dataset_fn_kwargs": self.__dataset_fn_kwargs,
        }
    
    def __str__(self):
        output_str = "Evaluator: \n"
        output_str += f"dataset_name: {self.dataset_name}\n"
        output_str += f"dataset_fn: {self._dataset_fn.__name__}\n"
        output_str += f"dataset_fn_kwargs: {self.__dataset_fn_kwargs}\n"
        # output_str += f"dataset_metadata: {self._dataset_metadata}\n"
        output_str += f"default evaluation functions: {pprint.pformat(list(self._default_eval_functions.keys()))}\n"
        output_str += f"custom evaluation functions: {pprint.pformat(list(self._custom_eval_functions.keys()))}\n"
        
        return output_str
    
    def set_dataset_fn(self, dataset_name: str=None, dataset_fn: callable=None) -> None:
        """Set dataset function

        Args:
            dataset_name (str): Name of dataset
            dataset_fn (callable): A function which returns a dataset. 
                                    dataset_fn must return a dict of 
                                    {"train": {"X": dataset, "y": target, "metainfo": metainfo}, 
                                    "test": {"X": dataset, "y": target, "metainfo": metainfo}}
        """
        if (dataset_name is None and dataset_fn is None) or (dataset_name is not None and dataset_fn is not None):
            raise ValueError("Either dataset_name or dataset_fn must be specified")

        if dataset_fn is not None:
            self._dataset_fn = dataset_fn
            self.dataset_name = dataset_fn.__name__
            return
        
        self._dataset_fn = utils.get_dataset_fn(dataset_name)
        self.dataset_name = dataset_name
        for eval_fn_name in const.DATASET2DEFAULT_EVALS[dataset_name]:
            self._default_eval_functions[eval_fn_name] = utils.get_metric(eval_fn_name)
    
    def get_dataset(self, *args, **kwargs):
        """Get dataset
        args and kwargs are passed to dataset_fn. See dataset_fn for more details
        If some keyword arguments are already set in the dataset_fn_kwargs, they are overwritten by kwargs.

        Returns:
            Dataset: A dataset
        """
        self.__dataset_fn_kwargs.update(kwargs)
        self._dataset_fn_ = utils.return_with_kwargs(self._dataset_fn) # Correspond to decorator
        self.__dataset_fn_kwargs, datasets = self._dataset_fn_(*args, **self.__dataset_fn_kwargs)
        self.__dataset_metadata = datasets["test"].get("metainfo", None)
        self.__target = datasets["test"]["y"]

        return datasets["test"]["X"]
    
    def add_custom_eval_fn(self, function_name: str, function: callable, **kwargs) -> None:
        """Add custom evaluation functions

        Args:
            function_name (str): Name of evaluation function
            function (callable): A callable functions. 
                                This function must take preds, ground_truths, metainfo and 
                                return a dictionary of evaluation results.
        """
        if len(kwargs) > 0:
            function = functools.partial(function, **kwargs)
        self._custom_eval_functions[function_name] = function

    def evaluate(self, preds: torch.Tensor) -> None:
        """Evaluate predictions

        Args:
            preds (torch.Tensor): A tensor of predictions
        """
        now_str = datetime.now().strftime("%Y%m%d%H%M%S")
        
        reports = OrderedDict()
        reports["metainfo"] = OrderedDict(
            date=now_str,
            dataset_name=self.dataset_name,
            dataset_kwargs=self.__dataset_fn_kwargs,
            dataset_metadata=self.__dataset_metadata,
        )
        # TODO: 辞書のupdateを使うか, それとも辞書の中に辞書を入れるかを検討する
        for fn_name, fn in self._default_eval_functions.items():
            reports[fn_name] = fn(preds, self.__target, self.__dataset_metadata)
        
        for fn_name, fn in self._custom_eval_functions.items():
            reports[fn_name] = fn(preds, self.__target, self.__dataset_metadata)
        
        return reports
    
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
        