# -- coding: utf-8 --

from addict import Dict

# @Time : 2022/9/12 11:52
# @Author : Zhiheng Feng
# @File : cfg_psf_train.py
# @Software : PyCharm


config = Dict()
config.exp_name = 'PSFNet'
config.SEED = 3407
config.model_option = {
    'type': "PoseSenceFlowModule",
    'is_training': True
}

MAX_EPOCH = 800
config.train_option = {
    'devices': ['cuda:0'],  # choose gpus or cpu, for example ['cuda:0', 'cuda:1']
    'resume_from': '',

    'optimizer': {'type': 'Adam', 'lr': 0.001, 'weight_decay': 1e-4},
    'scheduler': {'type': 'StepLR', 'step_size': 13, 'gamma': 0.7, 'last_epoch': -1},
    'learning_rate_clip': 1e-5,

    'epochs': MAX_EPOCH,
    'print_interval': 1,  # step为单位
    'val_interval': 1,  # epoch为单位

    # 拼写具体的名称，模块前加'-'减号表示不训练, 支持'all'表示全部训练, 例如['all'， '-backbone']表示除backbone外全部训练
    'train_stage': [['backbone', 'ocr_det_neck', 'ocr_det_head'], ['all', '-backbone']],
    # 注意，每个分界点是包含在左侧阶段的。即训练阶段为(train_step[i], train_step[i+1]]
    'train_step': [4, MAX_EPOCH],

    'ckpt_save_dir': f"./output/{config.exp_name}/checkpoint",  # 模型保存地址，log文件也保存在这里
    'ckpt_save_type': 'HighestAcc',  # HighestAcc：只保存最高准确率模型 ；FixedEpochStep：每隔ckpt_save_epoch个epoch保存一个
    'ckpt_save_epoch': 4,  # epoch为单位, 只有ckpt_save_type选择FixedEpochStep时，该参数才有效

}

config.dataset_option = {
    'train': {
        'kod': {
            'dataset': {  # 元数据在雷达坐标系, 已在内部完成从雷达坐标系到相机坐标系的变换，然后再做的pre_processes
                'type': 'KOD',
                'dataset_path': '/dataset/data_odometry_velodyne/dataset',
                'seqs_list': [4],
                'check_seq_len': True,
                'pre_processes': [{'type': 'RangeLimit', 'args': {'x_range': [-30, 30], 'y_range': [-1, 1.4],
                                                                  'z_range': [0, 35]}},
                                  {'type': 'RandomSample', 'args': {'num_points': 8192, 'allow_less_points': False}},
                                  {'type': 'ShakeAug', 'args': {'x_clip': 0.02, 'y_clip': 0.1, 'z_clip': 0.02}}]
            },
            'loader': {
                'type': 'DataLoader',
                'batch_size': 8,
                'shuffle': True,
                'num_workers': 0,
            }
        }
    },
    'sf_eval': {
        'ksf142': {  # 元数据, z指向车前，x指向左侧，y指向上方
            'dataset': {
                'type': 'KSF142',
                'dataset_path': '/dataset/KITTI_processed_occ_final',
                'just_eval': True,
                'train': False,
                'pre_processes': [{'type': 'RangeLimitWithSF', 'args': {'x_range': None, 'y_range': [-1.4, 1e3],
                                                                        'z_range': [0, 35]}, 'use_same_indice': True},
                                  {'type': 'RandomSampleWithSF',
                                   'args': {'num_points': 8192, 'allow_less_points': False}, 'no_corr': True},
                                  {'type': 'TransformerWithSF', 'args': {
                                      'Tr': [[-1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]}}]
            },
            'loader': {
                'type': 'DataLoader',
                'batch_size': 1,
                'shuffle': False,
                'num_workers': 0,
            },
            'eval_2d': True
        },
        'ksf150': {  # 元数据, 雷达坐标系, 已在内部完成从雷达坐标系到相机坐标系的变换，然后再做的pre_processes
            'dataset': {
                'type': 'KSF150',
                'dataset_path': '/dataset/kitti_rm_ground',
                'just_eval': True,
                'train': False,
                'pre_processes': [{'type': 'RangeLimitWithSF', 'args': {'x_range': None, 'y_range': [-1.4, 1e3],
                                                                        'z_range': [0, 35]}},
                                  {'type': 'RandomSampleWithSF',
                                   'args': {'num_points': 8192, 'allow_less_points': False, 'no_corr': True}}
                                  ]
            },
            'loader': {
                'type': 'DataLoader',
                'batch_size': 1,
                'shuffle': False,
                'num_workers': 0,
            },
            'eval_2d': False
        },
        'lidarKITTI': {  # 元数据, z指向车前，x指向左侧，y指向上方
            'dataset': {
                'type': 'lidarKITTI',
                'dataset_path': '/dataset/lidarKITTI',
                'just_eval': True,
                'train': False,
                'pre_processes': [{'type': 'RangeLimitWithSF', 'args': {'x_range': None, 'y_range': [-1.4, 1e3],
                                                                        'z_range': [0, 35]}},
                                  {'type': 'RandomSampleWithSF',
                                   'args': {'num_points': 8192, 'allow_less_points': False, 'no_corr': True}},
                                  {'type': 'TransformerWithSF', 'args': {
                                      'Tr': [[-1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]}}
                                  ]
            },
            'loader': {
                'type': 'DataLoader',
                'batch_size': 1,
                'shuffle': False,
                'num_workers': 0,
            },
            'eval_2d': False
        }
    },
    'pose_eval': {
        'seqs_list': [4],  # 评估哪些序列
        'kod': {  # 元数据, z指向车前，x指向左侧，y指向上方
            'dataset': {
                'type': 'KOD',
                'dataset_path': '/dataset/data_odometry_velodyne/dataset',
                'check_seq_len': True,
                'pre_processes': [{'type': 'RangeLimit', 'args': {'x_range': [-30, 30], 'y_range': [-1, 1.4],
                                                                  'z_range': [0, 35]}},
                                  {'type': 'RandomSample', 'args': {'num_points': 8192, 'allow_less_points': False}}]
            },
            'loader': {
                'type': 'DataLoader',
                'batch_size': 1,
                'shuffle': False,
                'num_workers': 0,
            }
        }
    }
}

# 转换为 Dict
for k, v in config.items():
    if isinstance(v, dict):
        config[k] = Dict(v)

if __name__ == '__main__':
    pass
