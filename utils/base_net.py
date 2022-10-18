# -- coding: utf-8 --
import os
import copy
import torch
import logging

from addict import Dict

# @Time : 2022/9/27 22:00
# @Author : Zhiheng Feng
# @File : base_net.py
# @Software : PyCharm

"""
    BaseNet 是模型工厂的基类，可以自动完成optimizer, scheduler, 模型加载、移动到GPU、保存等功能，
"""


class BaseNet(object):
    def __init__(self, model, train_config: Dict, logger: logging.Logger):
        """

        Args:
            model: 工厂需要处理的模型
            train_config: 训练设置，提供以下键devices，optimizer，scheduler，resume_from
            logger: 日志文件
        """
        super().__init__()
        self.net = model
        self.logger = logger
        self.devices = self.get_devices_ids(train_config.devices)

        self.logger.info(f'Number of {self.net.__class__.__name__} parameters: {self.get_para_num()}')
        self.optimizer = self.creat_optimizer(train_config.optimizer)
        self.scheduler = self.creat_scheduler(train_config.scheduler)
        self.global_state = {}

        self.load_model(train_config.resume_from)
        self.move_model(self.devices)

    def get_devices_ids(self, devices_config) -> list:
        """

        Args:
            devices_config: 传入的device配置参数，如['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

        Returns:
            将输入的字符串设备转换成torch.device的列表进行返回
            for example:输入为['cuda:0'] 返回为[device(type='cuda:0')]

        Raises:
            ValueError:输入的列表元素不是'cpu'或者'cuda:0

        """
        example = f"您输入的devices参数为{devices_config}，请输入正确的devices配置参数如 ['cpu']或['cuda:0', cuda:1]"
        devices_config = list(devices_config)
        if (len(devices_config) == 0):
            self.logger.error(example)
            raise ValueError(example)
        if not 'cuda' in devices_config[0] and not 'cpu' in devices_config[0]:
            self.logger.error(example)
            raise ValueError(example)
        if not torch.cuda.is_available() or devices_config[0] == 'cpu':
            return [torch.device('cpu')]
        else:
            return [torch.device(x) for x in devices_config]

    def move_model(self, devices: list):
        """

        Args:
            devices: 传入一个torch.device的列表

        Returns:
            根据devices，将模型放到对应的设备上

        """
        if len(devices) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=devices)
        self.net.to(devices[0])
        if not self.optimizer is None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(devices[0])
        self.logger.info(f'{self.net.__class__.__name__} trained on : {devices}')

    def load_model(self, ckpt_path: str):
        """

        Args:
            ckpt_path: 预加载模型路径

        Returns:
            加载模型

        """
        self.global_state = {}
        if ckpt_path:
            if os.path.isfile(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=self.devices[0])
                self.net.load_state_dict(self.strip_prefix(checkpoint['model_state_dict']))
                self.logger.info(f'{self.net.__class__.__name__} loads pretrain model {ckpt_path}!')
                if 'optimizer' in checkpoint.keys():
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                    self.logger.info(f'Optimizer loads pretrain from {ckpt_path}!')
                if 'scheduler' in checkpoint.keys():
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    self.logger.info(f'Scheduler loads pretrain from {ckpt_path}!')
                self.global_state = checkpoint['global_state']
            else:
                self.logger.warning(f'{ckpt_path} is not is file, train from scratch')
        else:
            self.logger.warning('train from scratch')

    def creat_optimizer(self, config):
        """

        Args:
            config:

        Returns:

        """
        if config:
            optimizer_config = copy.deepcopy(config)
            optimizer_type = optimizer_config.pop('type')
            optimizer = eval('torch.optim.{}'.format(optimizer_type))(self.net.parameters(), **optimizer_config)
            self.logger.info(f'the optimizer of your model {self.net.__class__.__name__} is {optimizer}!')
            return optimizer
        else:
            self.logger.warning(f'your model {self.net.__class__.__name__} has no optimizer!')
            return None

    def creat_scheduler(self, config):
        """

        Returns: 创建一个scheduler

        """
        if config:
            scheduler_config = copy.deepcopy(config)
            scheduler_type = scheduler_config.pop('type')
            scheduler = eval('torch.optim.lr_scheduler.{}'.format(scheduler_type))(self.optimizer, **scheduler_config)
            self.logger.info(f'the scheduler of your model {self.net.__class__.__name__} is {scheduler}!')
            return scheduler
        else:
            self.logger.warning(f'your model {self.net.__class__.__name__} has no scheduler!')
            return None

    def get_para_num(self):
        """

        Returns:
            获取模型中参数的数量

        """
        return sum([p.data.nelement() for p in self.net.parameters()])

    def get_learing_rate(self) ->str:
        """

        Returns:
            返回优化器中各参数组的学习率

        """

        lr_str = 'param_group '
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr_str += f"{i}-th {param_group['lr']}"
        return lr_str

    def save_checkpoint(self, checkpoint_path, **kwargs):
        """

        Args:
            checkpoint_path: 模型保存路径
            **kwargs:

        Returns: 保存最新的模型和ckpt_save_type决定的模型

        """
        save_state = {
            'model_state_dict': self.net.module.state_dict() if hasattr(self.net, 'module') else self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        save_state.update(kwargs)  # 用于保存模型所处的全局环境之类的变量
        torch.save(save_state, checkpoint_path)
        self.logger.info(f'Save {self.net.__class__.__name__} to {checkpoint_path}')

    def strip_prefix(self, state_dict: dict, prefix: str = 'module.') -> dict:
        """
        load_model时会自动调用
        :param state_dict: 存储的模型字典
        :param prefix: 前缀
        :return: 除去前缀后的模型字典，该方法用于将模型加载到单卡时的处理操作
        """
        if not all(key.startswith(prefix) for key in state_dict.keys()):
            return state_dict
        stripped_state_dict = {}
        for key in list(state_dict.keys()):
            stripped_state_dict[key[len(prefix):]] = state_dict.pop(key)
        return stripped_state_dict

    def add_prefix(self, state_dict: dict, prefix: str = 'module.') -> dict:
        """
        load_model时会自动调用
        :param state_dict: 存储的模型字典
        :param prefix: 前缀
        :return: 加上前缀后的模型字典，该方法用于将模型加载到多卡时的处理操作
        """
        if all(key.startswith(prefix) for key in state_dict.keys()):
            return state_dict
        stripped_state_dict = {}
        for key in list(state_dict.keys()):
            key2 = prefix + key
            stripped_state_dict[key2] = state_dict.pop(key)
        return stripped_state_dict


