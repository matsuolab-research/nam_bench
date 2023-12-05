import inspect
import dataclasses
from dataclasses import dataclass, field
from functools import partial

import nam_bench.metrics as custom_metrics
from nam_bench import datasets


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


def return_with_kwargs(func: callable) -> callable:
    """Decorator to return a function with kwargs

    Args:
        func (callable): A function to be decorated

    Returns:
        callable: Wrapped function
    """
    def wrapper(*args, **kwargs):
        args_count = len(args)
        fn_params = inspect.signature(func).parameters
        
        for arg, param_name in zip(args, list(fn_params.keys())[:args_count]):
            kwargs[param_name] = arg
        
        return kwargs, func(**kwargs)

    return wrapper


def get_metric(metric_name: str) -> callable:
    """Get callable metric

    Args:
        metric_name (str): Name of callable metric

    Raises:
        ValueError: Invalid metric name

    Returns:
        callable: A callable metric
    """
    try:
        metric = getattr(custom_metrics, metric_name)
    except:
        raise ValueError("Invalid metric name: {}".format(metric_name))
    
    return metric


def get_metrics(metrics: list[str]) -> dict[str, callable]:
    """Get callable metrics

    Args:
        metrics[list]: str list of callable metrics.

    Raises:
        ValueError: Invalid metrics name

    Returns:
        dict[callable]: A dictionary of callable metrics
    """
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric] = get_metric(metric)
    
    return metrics_dict


def get_dataset_fn(dataset_name: str) -> callable:
    """Get dataset

    Args:
        dataset_name (str): Name of dataset

    Raises:
        ValueError: Invalid dataset name

    Returns:
        (callable): A function which returns a dataset. 
                    dataset_fn must return a dict of 
                    {"train": {"X": dataset, "y": target, "metainfo": metainfo}, 
                    "test": {"X": dataset, "y": target, "metainfo": metainfo}}
    """
    try:
        dataset_fn = datasets.NAME2DATASETS_FN[dataset_name]
    except:
        raise ValueError("Invalid dataset name: {}".format(dataset_name))
    
    return dataset_fn