# -- coding: utf-8 --
import torch
import torch.nn as nn


# @Time : 2022/10/14 15:33
# @Author : Zhiheng Feng
# @File : odometry_loss.py
# @Software : PyCharm


class OdometryLossModel(nn.Module):
    def __init__(self, weights: tuple = (0.2, 0.4, 0.8, 1.6)):
        super().__init__()
        self.weights = weights
        self.w_x = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.w_q = torch.nn.Parameter(torch.tensor([-2.5]), requires_grad=True)

    def forward(self, pre_q_list, pre_t_list, q_gt, t_gt):
        """

        Args:
            pre_q_list: coarse_to_fine预测的每一层的归一化的四元数,每个元素shape为[]
            pre_t_list: coarse_to_fine预测的每一层的位姿
            q_gt: 位姿变换的四元数真值
            t_gt: 位姿变换的平移真值

        Returns:位姿损失

        """
        loss_list = []

        for pre_q_norm, pre_t, weight in zip(pre_q_list, pre_t_list, self.weights):
            loss_q = torch.mean(
                torch.sqrt(torch.sum((q_gt - pre_q_norm) * (q_gt - pre_q_norm), dim=-1, keepdim=True) + 1e-10))
            loss_t = torch.mean(torch.sqrt((pre_t - t_gt) * (pre_t - t_gt) + 1e-10))
            loss = loss_t * torch.exp(-self.w_x) + self.w_x + loss_q * torch.exp(-self.w_q) + self.w_q
            loss_list.append(weight * loss)
        return sum(loss_list)
