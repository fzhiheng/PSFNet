# -- coding: utf-8 --
import copy

from .PSFNetModel import PoseSenceFlowModule

# @Time : 2022/10/14 12:09
# @Author : Zhiheng Feng
# @File : __init__.py
# @Software : PyCharm

support_model = {'PoseSenceFlowModule': PoseSenceFlowModule}


__all__ = ['build_model']

def build_model(model_config):
    copy_config = copy.deepcopy(model_config)
    model_type = copy_config.pop('type')
    assert model_type in support_model, f'model_type {model_type} must in {support_model.keys()}'
    model = support_model[model_type](**copy_config)
    return model