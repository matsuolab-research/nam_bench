import dataclasses
from dataclasses import dataclass, field
from functools import partial

from torch.utils.data import Dataset

import src.metrics as custom_metrics
from src import datasets, callbacks


def rec_getattr(obj, attr):
    """Get object's attribute. May use dot notation.

    >>> class C(object): pass
    >>> a = C()
    >>> a.b = C()
    >>> a.b.c = 4
    >>> rec_getattr(a, 'b.c')
    4
    """
    if '.' not in attr:
        return getattr(obj, attr)
    else:
        L = attr.split('.')
        return rec_getattr(getattr(obj, L[0]), '.'.join(L[1:]))


def rec_setattr(obj, attr, value):
    """Set object's attribute. May use dot notation.

    >>> class C(object): pass
    >>> a = C()
    >>> a.b = C()
    >>> a.b.c = 4
    >>> rec_setattr(a, 'b.c', 2)
    >>> a.b.c
    2
    """
    if '.' not in attr:
        setattr(obj, attr, value)
    else:
        L = attr.split('.')
        rec_setattr(getattr(obj, L[0]), '.'.join(L[1:]), value)


@dataclass
class EvaluationDataset():
    name: str
    data: Dataset
    metainfo: dict = field(default=None)


class Evaluator:
    def __init__(self, metrics_dict: dict[callable]):
        self.metrics_dict = metrics_dict
    
    def __call__(self, preds, ground_truth, dataset_metainfo=None): # TODO: Wrap metrics to take dataset_metainfo
        reports = {}
        for metric_name, metric in self.metrics_dict.items():
            reports.update(metric(preds, ground_truth, dataset_metainfo))
        
        return reports


def get_metrics(metrics: list[str]) -> dict[callable]:
    """Get callable metrics

    Args:
        metrics[list[dict]]: list of callable metrics in str.

    Raises:
        ValueError: Invalid metrics name

    Returns:
        dict[callable]: A dictionary of callable metrics
    """
    metrics_dict = {}
    for metric in metrics:
        try:
            metrics_dict[metric] = getattr(custom_metrics, metric)
        except:
            raise ValueError("Invalid metrics name: {}".format(metric))
    
    return metrics_dict


def get_dataset(config: dict) -> EvaluationDataset:
    """Get dataset

    Args:
        config (dict): configuration for evaluation

    Raises:
        ValueError: Invalid dataset name

    Returns:
        EvaluationDataset: Evaluation dataset
    """
    try:
        tmp_dataset = getattr(datasets, config.dataset_name)
    except:
        raise ValueError("Invalid dataset name: {}".format(config.dataset_name))
    
    tmp_dataset = tmp_dataset(config)
    eval_datasets = EvaluationDataset(
        name=config.dataset_name,
        data=tmp_dataset,
        metainfo=tmp_dataset.metainfo,
    )
    
    return eval_datasets


def get_callbacks(callbacks_config: object) -> list[callable]:
    """Get callbacks

    Args:
        callbacks_config (object): A configuration of callbacks. 
        The key is the name of callback and the value is the configuration for callback.

    Returns:
        dict[callable]: A dictionary of callbacks
    """
    callbacks_ = []
    for callback_name, callback_config in vars(callbacks_config).items():
        try:
            callbacks_.append(partial(getattr(callbacks, callback_name), **vars(callback_config)))
        except:
            raise ValueError("Invalid callback name: {}".format(callback_name))
    
    return callbacks_