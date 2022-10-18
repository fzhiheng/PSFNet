# -- coding: utf-8 --

import copy

from addict import Dict
from torch.utils.data import DataLoader

from .kod import KOD
from .ksf142 import KSF142
from .ksf150 import KSF150
from .lidarKITTI import lidarKITTI

# @Time : 2022/10/13 21:56
# @Author : Zhiheng Feng
# @File : __init__.py
# @Software : PyCharm


support_dataset_set = {'KOD', 'KSF142', 'KSF150', 'lidarKITTI'}
support_loader = {'DataLoader'}

def build_dataset(config):
    """

    Args:
        config: 数据集相关的配置，必须包含'type'，如: 'KSF142',

    Returns:根据配置构造好的 DataSet 类对象

    Raises:
        参数中dataset的类型不受支持的时候抛出ValueError错误

    """
    copy_config = copy.deepcopy(config)
    dataset_type = copy_config.pop('type')
    if not dataset_type in support_dataset_set:
        raise ValueError(f'{dataset_type} is not developed yet!, only {support_dataset_set} are support now')
    dataset = eval(dataset_type)(**copy_config)
    return dataset


def build_loader(dataset, config):
    """

    Args:
        dataset: 继承自 torch.utils.data.DataSet的类对象
        config: loader 相关的配置，一般包含'type': 'DataLoader',batch_size, shuffle,num_workers等等

    Returns:根据配置构造好的 DataLoader 类对象

    """
    dataloader_type = config.pop('type')
    if not dataloader_type in support_loader:
        raise ValueError(f'{dataloader_type} is not developed yet!, only {support_loader} are support now')

    # build collate_fn
    if 'collate_fn' in config:
        config['collate_fn']['dataset'] = dataset
        collate_fn = build_collate_fn(config.pop('collate_fn'))
    else:
        collate_fn = None
    dataloader = eval(dataloader_type)(dataset=dataset, collate_fn=collate_fn, **config ,pin_memory=True)
    return dataloader


def build_collate_fn(config):
    """

    Args:
        config: collate_fn 相关的配置

    Returns:    :return: 根据配置构造好的 collate_fn 类对象


    """
    collate_fn_type = config.pop('type')
    if len(collate_fn_type) == 0:
        return None
    collate_fn_class = eval(collate_fn_type)(**config)
    return collate_fn_class


def build_dataloader(config):
    """
    根据配置构造 dataloader, 包含两个步骤，1. 构造 dataset, 2. 构造 dataloader
    Args:
        config: 数据集相关的配置,包含dataset和loader两个键值

    Returns:根据配置构造好的 DataLoader 类对象

    """

    # build dataset
    copy_config = copy.deepcopy(config)
    copy_config = Dict(copy_config)
    dataset = build_dataset(copy_config.dataset)

    # build loader
    loader = build_loader(dataset, copy_config.loader)
    return loader
