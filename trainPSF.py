# -*- coding:UTF-8 -*-

import os
import copy
import torch
import random
import shutil
import traceback
import numpy as np
import pandas as pd

from network.psf_net import PSFNet
from dataset import build_dataloader
from utils.AverageMeter import AverageMeter
from utils.logging_utils import get_logger
from utils.config_utils import parse_args
from network.loss.odometry_loss import OdometryLossModel
from network.loss.scene_flow_loss import SceneFlowLossModel

# author:Zhiheng Feng
# datetime:2022/10/13 15:23
# software: PyCharm


def set_random_seed(seed=3407, use_cuda=True, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def main():
    # =========> 获取配置文件参数
    cfg = parse_args()

    # =========> 创建结果保存路径
    ckpt_path = cfg.train_option.ckpt_save_dir
    os.makedirs(ckpt_path, exist_ok=True)
    # if os.path.exists(ckpt_path):
    #     raise FileExistsError(f"{ckpt_path} already exists")
    # else:
    #     os.makedirs(ckpt_path)

    # =========> 创建日志文件
    logger = get_logger('psfnet', log_file=os.path.join(ckpt_path, 'train.log'))
    logger.info(cfg)

    # ======> 固定随机种子
    set_random_seed(seed=cfg.SEED, use_cuda='cuda' in cfg.train_options.devices[0] and torch.cuda.is_available(),
                    deterministic=True)

    # ======> 构建训练模型
    psf_net = PSFNet(cfg.model_option, cfg.train_option, logger)

    # =========> 创建损失函数
    pose_loss_func = OdometryLossModel()
    pose_loss_func.to(psf_net.devices[0])

    sf_loss_func = SceneFlowLossModel()
    sf_loss_func.to(psf_net.devices[0])

    # =========> 创建训练数据
    kod_train_loader = build_dataloader(cfg.dataset_option.train.kod)
    all_step = len(kod_train_loader)
    logger.info(f'train dataset has {kod_train_loader.dataset.__len__()} samples,{all_step} in dataloader')

    # =========> 创建场景流评估数据， ['lidarKITTI', 'ksf142', 'ksf150']
    sf_eval_dataset_name = list(cfg.dataset_option.sf_eval.keys())
    sf_eval_dataset_dict = dict()
    for key_name in sf_eval_dataset_name:
        data_loader = build_dataloader(cfg.dataset_option.sf_eval[key_name])
        sf_eval_dataset_dict[key_name] = dict()
        sf_eval_dataset_dict[key_name]['data_loader'] = data_loader
        sf_eval_dataset_dict[key_name]['eval_2d'] = cfg.dataset_option.sf_eval[key_name]['eval_2d']
        logger.info(
            f'{key_name} eval dataset has {data_loader.dataset.__len__()} samples,{len(data_loader)} in dataloader')
        if sf_eval_dataset_dict[key_name]['eval_2d']:
            key_list = ('EPE3D', 'ACC3DS', 'ACC3DR', 'Outliers3D', 'EPE2D', 'ACC2D')
        else:
            key_list = ('EPE3D', 'ACC3DS', 'ACC3DR', 'Outliers3D')
        sf_eval_dataset_dict[key_name]['key_list'] = key_list
        metrics = pd.DataFrame(columns=key_list, dtype=float)
        metrics.index.name = 'epoch'
        sf_eval_dataset_dict[key_name]['metrics'] = metrics

    # =========> 创建里程计评估数据 依序列为单位
    pose_eval_seq = cfg.dataset_option.pose_eval.seqs_list
    pose_eval_dataset_dict = dict()
    for seq in pose_eval_seq:
        key_name = f'{seq:02d}'
        dataset_option = copy.deepcopy(cfg.dataset_option.pose_eval.kod)
        dataset_option.dataset.update({'seqs_list': [seq]})
        data_loader = build_dataloader(dataset_option)
        pose_eval_dataset_dict[key_name] = dict()
        pose_eval_dataset_dict[key_name]['data_loader'] = data_loader
        logger.info(
            f'seq {key_name} eval dataset has {data_loader.dataset.__len__()} samples,{len(data_loader)} in dataloader')
        key_list = ['rotate', 'trans']
        pose_eval_dataset_dict[key_name]['key_list'] = key_list
        metrics = pd.DataFrame(columns=key_list, dtype=float)
        metrics.index.name = 'epoch'
        pose_eval_dataset_dict[key_name]['metrics'] = metrics

    # =========> 获取全局状态记录
    global_state = psf_net.global_state
    start_epoch = global_state.setdefault('start_epoch', 0)
    best_model = global_state.setdefault('best_model', {'best_epoch': 0})

    # =========> 创建结果保存路径
    pose_pred_root = os.path.join(ckpt_path, 'pose_pred')
    pose_visual_root = os.path.join(ckpt_path, 'pose_visual')
    pose_xlsx_root = os.path.join(ckpt_path, 'pose_xlsx')
    sf_xlsx_root = os.path.join(ckpt_path, 'sf_xlsx')

    # =========> 创建计时模块，估计训练耗时
    eta_meter = AverageMeter()
    try:
        # =========> 开始训练
        for epoch in range(start_epoch, psf_net.max_epoch + 1):

            psf_net.train(kod_train_loader, sf_loss_func, pose_loss_func, eta_meter, epoch)

            if (epoch + 1) % cfg.train_option.val_interval == 0:
                global_state['start_epoch'] = epoch
                global_state['best_model'] = best_model
                net_save_path = f"{ckpt_path}/latest.pth.tar"
                psf_net.save_checkpoint(net_save_path, global_state=global_state)
                # 里程计指标评估与场景流指标评估
                pose_eval_result = psf_net.eval_pose(pose_eval_dataset_dict, epoch, pose_pred_root, pose_visual_root,
                                                     pose_xlsx_root)
                sf_eval_result = psf_net.eval_sf(sf_eval_dataset_dict, epoch, sf_xlsx_root)

                if cfg.train_option.ckpt_save_type == 'HighestAcc':
                    pass
                    # if eval_dict['hmean'] > best_model['hmean']:
                    #     best_model.update(eval_dict)
                    #     best_model['best_model_epoch'] = epoch
                    #     best_model['models'] = net_save_path
                    #     global_state['start_epoch'] = epoch
                    #     global_state['best_model'] = best_model
                    #     net_save_path = f"{ckpt_path}/best.pth.tar"
                    #     psf_net.save_checkpoint(net_save_path, global_state=global_state)
                elif cfg.train_option.ckpt_save_type == 'FixedEpochStep' and epoch % cfg.train_option.ckpt_save_epoch == 0:
                    shutil.copy(net_save_path, net_save_path.replace('latest.pth.tar', f'{epoch}.pth.tar'))
                best_str = 'current best, '
                for k, v in best_model.items():
                    best_str += '{}: {}, '.format(k, v)
                logger.info(best_str)
    except KeyboardInterrupt:
        net_save_path = f"{ckpt_path}/final.pth.tar"
        psf_net.save_checkpoint(net_save_path, global_state=global_state)
    except:
        error_msg = traceback.format_exc()
        logger.error(error_msg)
    finally:
        for k, v in best_model.items():
            logger.info(f'{k}: {v}')


if __name__ == '__main__':
    main()
