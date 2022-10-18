# -*- coding:UTF-8 -*-

import os
import time
import torch
import logging
import datetime
import numpy as np
import pandas as pd

from addict import Dict
from tqdm import tqdm

from network.model import build_model
from utils.base_net import BaseNet
from utils.euler_tools import quat2mat
from utils.AverageMeter import AverageMeter
from utils.metric_analysis import odo_metric_analysis
from network.metric.scene_flow.geometry import get_batch_2d_flow
from network.metric.scene_flow.sf_metric import evaluate_2d, evaluate_3d
from network.metric.odometry.odometry_metric import kittiOdomEval

# author:Zhiheng Feng
# datetime:2022/10/15 10:46
# software: PyCharm

"""
文件说明：
此时的psf.net中的flow场景流采取的损失已经是新的损失了，在原损失的基础上加上了OG提出的occ_mask和flow_self损失
后面所进行的一系列对比实验都是在这个新的损失的前提下进行的，注意这一点就可以了

"""


def dict2list(input_dict: dict, key_order: tuple) -> list:
    """
    将字典的key-value，key按照key_order的顺序，将value保存到一个列表中输出
    Args:
        input_dict: 输入的字典
        key_order: key的顺序

    Returns: value保存到一个列表中输出

    """
    return [input_dict[k] for k in key_order]


class PSFNet(BaseNet):

    def __init__(self, model_config: Dict, train_config: Dict, logger: logging.Logger):
        psf_model = build_model(model_config)
        super().__init__(psf_model, train_config, logger)
        self.lr_clip = train_config.learning_rate_clip
        self.print_interval = train_config.print_interval
        self.max_epoch = train_config.epochs

    # TODO 分阶段训练功能尚未添加 @fzh
    def train(self, train_loader, sf_loss_func, pose_loss_func, eta_meter, epoch):
        """

        Args:
            train_loader: 训练数据集
            sf_loss_func: 场景流损失函数
            pose_loss_func: 里程计损失函数
            eta_meter: 计时模块
            epoch: 当前训练epoch

        Returns:

        """
        self.net = self.net.train()
        start = time.time()
        odometry_loss = 0
        sf_loss = 0
        all_step = len(train_loader)
        global_step = len(train_loader) * epoch
        lr = max(self.optimizer.param_groups[0]['lr'], self.lr_clip)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        for i, batch_data in enumerate(train_loader):
            for key, value in batch_data.items():
                if value is not None and isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(self.devices[0])

            pre_q_list, pre_t_list, pc1_sample_2048, pc1, pc2, pred_flows, occ_masks, static_masks, label = self.net(
                batch_data['lidar1'], batch_data['lidar2'], batch_data['norm1'], batch_data['norm2'], state='all')

            self.optimizer.zero_grad()
            pose_loss_func.train()
            pose_loss = pose_loss_func(pre_q_list, pre_t_list, batch_data['q_gt'], batch_data['t_gt'])
            sf_loss = sf_loss_func(pc1, pc2, pred_flows)
            loss = pose_loss + sf_loss
            loss.requires_grad_(True)
            loss.backward()
            self.optimizer.step()
            odometry_loss += pose_loss.item()
            sf_loss += sf_loss.item()
            train_batch_time = float(time.time() - start)
            eta_meter.update(train_batch_time)

            # ----------> 打印出相关信息
            if (i + 1) % self.print_interval == 0:
                eta_sec = ((self.max_epoch + 1 - epoch) * all_step - i - 1) * eta_meter.avg
                eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                self.logger.info(f"epoch[{epoch}/{self.max_epoch}] - "
                                 f"step[{i + 1}/{all_step}] - "
                                 f"lr:{self.get_learing_rate()} - "
                                 f"odometry loss:{pose_loss.item()} - "
                                 f"sf loss:{sf_loss.item()} - "
                                 f"time:{train_batch_time:.4f} - "
                                 f"eta: {eta_sec_format}")
                start = time.time()
            global_step += 1

        self.logger.info(f"odometry train_loss: {odometry_loss / all_step}")
        self.logger.info(f"scene flow train_loss: {sf_loss / all_step}")
        self.scheduler.step()

    def eval_pose(self, eval_data_dict, cur_epoch, pred_save_root, visual_save_root, xlsx_save_root):
        """

        Args:
            eval_data_dict: 每个键值对中包含loader,metric，key_list信息
            cur_epoch: 当前评估的epoch
            pred_save_path: 推理结果保存根路径，将以npy数据进行保存
            visual_save_path: 可视化结果保存根路径，将以pdf数据进行保存
            xlsx_save_path: excel分析结果保存根路径，将以xlsx数据进行保存

        Returns: 返回一个字典，包含每个评估序列的结果

        """
        epoch = cur_epoch
        null = list(map(self.check_path, [pred_save_root, visual_save_root, xlsx_save_root]))

        # 对每个序列进行推理
        # print(eval_data_dict)
        for seq, value in eval_data_dict.items():
            self.logger.info(f" start to eval odometry dataset: {seq}")
            pred_save_path = os.path.join(pred_save_root, f'{seq}_pred.npy')
            self.eval_pose_one_seq(value['data_loader'], pred_save_path, eval_state='all')

        # 构建里程计评估类
        metric_cfg = Dict()
        metric_cfg.gt_dir = 'data/odometry_gt'
        metric_cfg.pre_result_dir = pred_save_root
        metric_cfg.visual_save_root = visual_save_root
        metric_cfg.eva_seqs = [key for key in eval_data_dict.keys()]
        metric_cfg.epoch = cur_epoch
        odo_metric = kittiOdomEval(metric_cfg)
        ave_metric = odo_metric.eval(toCameraCoord=False)

        # 将每个序列的评估指标保存,将所有序列的指标放在一张表中
        metrics_list = []
        seq_list = []
        for seq, value in eval_data_dict.items():
            result_dict = ave_metric[seq]
            value['metrics'].loc[str(epoch)] = dict2list(result_dict, value['key_list'])
            metrics_list.append(value['metrics'])
            seq_list.append(seq)
            if xlsx_save_root:
                value['metrics'].to_excel(os.path.join(xlsx_save_root, f'{seq}.xlsx'))
        all_seqs_metrics = pd.concat(metrics_list, axis=1)
        all_seqs_metrics.columns = [f'{seq}_{metric_name}' for metric_name in eval_data_dict[seq]['key_list']
                                    for seq in seq_list]
        odo_metric_analysis(all_seqs_metrics, eval_data_dict[list(eval_data_dict.keys())[0]]['key_list'])
        all_seqs_metrics.to_excel(os.path.join(xlsx_save_root, 'all_seqs.xlsx'))
        # eval_data_dict['seqs_metrics'] = all_seqs_metrics

        return ave_metric

    def eval_sf(self, eval_data_dict, cur_epoch, xlsx_save_root) -> dict:
        """

        Args:
            eval_data_dict: 每个键值对中包含loader,metric，key_list信息
            cur_epoch: 当前评估的epoch
            xlsx_save_path: excel分析结果保存根路径，将以xlsx和csv数据进行保存

        Returns: 返回一个字典，包含每个评估序列的结果

        """
        epoch = cur_epoch
        result = dict()
        for name, value in eval_data_dict.items():
            self.logger.info(f" start to eval sf dataset: {name}")
            result_dict = self.eval_sf_one_loader(value['data_loader'], eval_2d_flag=value['eval_2d'])
            result[name] = result_dict
            value['metrics'].loc[str(epoch)] = dict2list(result_dict, value['key_list'])
            if xlsx_save_root:
                if not os.path.exists(xlsx_save_root):
                    os.makedirs(xlsx_save_root)
                value['metrics'].to_excel(os.path.join(xlsx_save_root, f'{name}.xlsx'))
        return result

    def eval_pose_one_seq(self, val_loader, save_pred_path, eval_state='all'):
        """
            将预测结果保存在save_pred_path路径中
        Args:
            val_loader: 需要进行评估的里程计数据集
            save_pred_path: 预估的里程计结果保存路径

        Returns:

        """
        self.net = self.net.eval()
        with torch.no_grad():
            line = 0
            for batch_data in tqdm(val_loader):
                for key, value in batch_data.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        batch_data[key] = value.to(self.devices[0])
                pre_q_list, pre_t_list, *rest = self.net(batch_data['lidar1'], batch_data['lidar2'],
                                                         batch_data['norm1'], batch_data['norm2'], state='all')

                pred_q = pre_q_list[0].cpu().numpy()  # b,4
                pred_t = pre_t_list[0].cpu().numpy()

                # 评估支持batch_size = pc1.shape[0] > 1的情况
                for qq, tt in zip(pred_q, pred_t):
                    qq = qq.reshape(4)
                    tt = tt.reshape(3, 1)
                    RR = quat2mat(qq)
                    TT = np.concatenate([np.concatenate([RR, tt], axis=-1), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
                    # 求逆的原因是，上面估计的是从第一帧电云到第二帧点云的位姿变换，求完逆之后才是相机位姿变换
                    TT = np.linalg.inv(TT)
                    if line == 0:
                        T_final = TT  # 4 4
                        T = T_final[:3, :]  # 3 4
                        T = T.reshape(1, 12)
                        line += 1
                    else:
                        T_final = np.matmul(T_final, TT)
                        T_current = T_final[:3, :]
                        T_current = T_current.reshape(1, 12)
                        T = np.append(T, T_current, axis=0)
            T = T.reshape(-1, 12)
            np.save(save_pred_path, T)

    def eval_sf_one_loader(self, val_loader, eval_2d_flag=False, eval_state='all') -> dict:
        """

        Args:
            val_loader: 需要进行评估的场景流数据集
            eval_2d_flag: 是否需要2d评估指标
            eval_state:

        Returns: 字典形式返回各项指标

        """
        self.net = self.net.eval()
        epe3ds = AverageMeter()
        acc3d_stricts = AverageMeter()
        acc3d_relaxs = AverageMeter()
        outliers = AverageMeter()
        if eval_2d_flag:
            epe2ds = AverageMeter()
            acc2ds = AverageMeter()

        with torch.no_grad():
            for batch_data in tqdm(val_loader):
                for key, value in batch_data.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        batch_data[key] = value.to(self.devices[0])
                pre_q_list, pre_t_list, pc1_sample_2048, pc1, pc2, pred_flows, _, _, label = self.net(
                    batch_data['pc1'], batch_data['pc2'], batch_data['norm1'], batch_data['norm2'],
                    label=batch_data['flow'], state='all')

                full_flow = pred_flows[0]
                sf_gt = label[0]
                pc1_ = pc1[0]

                pc1_np = pc1_.cpu().numpy()
                sf_np = sf_gt.cpu().numpy()
                pred_sf = full_flow.cpu().numpy()
                epe3d, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)
                epe3ds.update(epe3d)
                acc3d_stricts.update(acc3d_strict)
                acc3d_relaxs.update(acc3d_relax)
                outliers.update(outlier)

                if eval_2d_flag:
                    flow_pred, flow_gt = get_batch_2d_flow(pc1_np, pc1_np + sf_np, pc1_np + pred_sf, batch_data['path'])
                    epe2d, acc2d = evaluate_2d(flow_pred, flow_gt)
                    epe2ds.update(epe2d)
                    acc2ds.update(acc2d)

        if eval_2d_flag:
            return {'EPE3D': epe3ds.avg, 'ACC3DS': acc3d_stricts.avg, 'ACC3DR': acc3d_relaxs.avg,
                    'Outliers3D': outliers.avg, 'EPE2D': epe2ds.avg, 'ACC2D': acc2ds.avg}
        else:
            return {'EPE3D': epe3ds.avg, 'ACC3DS': acc3d_stricts.avg, 'ACC3DR': acc3d_relaxs.avg,
                    'Outliers3D': outliers.avg}

    def check_path(self, path: str):
        """
        检查path是否存在，若不存在则创建
        Args:
            path: 需要检查的路径
        """
        if not os.path.exists(path):
            os.makedirs(path)
            return True
        return True
