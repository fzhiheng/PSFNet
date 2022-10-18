# -- coding: utf-8 --
import os
import bisect
import yaml
import numpy as np
import torch.utils.data as data

from utils.json_process import json2dict
from utils.euler_tools import euler2quat, mat2euler
from dataset.kod_preprocess import RangeLimit, RandomSample, ShakeAug



# @Time : 2022/9/12 16:37
# @Author : Zhiheng Feng
# @File : kod.py
# @Software : PyCharm

class KOD(data.Dataset):

    def __init__(self, dataset_path, seqs_list, pre_processes, check_seq_len=True,
                 gt_root='data/odometry_gt_diff', vel2cam_root='data/vel_to_cam_Tr.json'):
        """

        Args:
            dataset_path:
            seqs_list: KITTI雷达点云数据集所使用的序列，比如[0,1,2,3]
            train:
            pre_processes:
            check_seq_len:
            gt_root:
        """
        super().__init__()

        self.dataset_root = dataset_path
        seqs_list = sorted(seqs_list)
        self.seqs_list = seqs_list
        self.gt_root = gt_root

        # 这里标注的是每个点云序列中点云帧的最大索引
        kITTI_seqs_ = [4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200]
        expected_seqs_len = np.array([kITTI_seqs_[seq_index] + 1 for seq_index in seqs_list])  # KKTTI提供的每个序列的长度
        get_seqs_len = np.array(self.get_seqs_len())  # 在给定的路径中找到的每个序列的长度


        if check_seq_len:
            wrong_index = [i for i in range(len(get_seqs_len)) if not expected_seqs_len[i] == get_seqs_len[i]]
            wrong_seq = [seqs_list[index] for index in wrong_index]
            wrong_seq_len = [get_seqs_len[index] for index in wrong_index]
            expect_wrong_seq_len = [expected_seqs_len[index] for index in wrong_index]
            if wrong_index:
                raise ValueError(
                    f'the num of sequences {wrong_seq} is {wrong_seq_len}, but expected {expect_wrong_seq_len}')

        self.seqs_len_cumsum = [0] + list(np.cumsum(get_seqs_len))

        # 获取雷达点云到相机点云的变换矩阵
        vel2cam_dict = json2dict(vel2cam_root)
        self.vel_to_cam_Tr = [np.array(vel2cam_dict[f'Tr_{seq_index:02d}']) for seq_index in seqs_list]

        # 构建点云预处理类
        self._init_pre_processes(pre_processes)

    def __len__(self):
        return self.seqs_len_cumsum[-1]


    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):

        # 第二帧点云恰好在分界点上的时候，说明使用的两帧点云是该序列的两个第0帧点云
        if index in self.seqs_len_cumsum:
            seq_index = self.seqs_len_cumsum.index(index)
            fn2 = 0
            fn1 = 0
        else:
            seq_index = bisect.bisect(self.seqs_len_cumsum, index) - 1
            fn2 = index - self.seqs_len_cumsum[seq_index]
            fn1 = fn2 - 1


        data_dict = dict()

        # 读取相邻两帧点云并转换到相机坐标系下
        lidar_path = os.path.join(self.dataset_root, f'{self.seqs_list[seq_index]:02d}', 'velodyne')
        fn1_dir = os.path.join(lidar_path, f'{fn1:06d}.bin')
        fn2_dir = os.path.join(lidar_path, f'{fn2:06d}.bin')
        lidar1, lidar2 = list(map(self.read_lidar, [fn1_dir, fn2_dir]))
        Tr = self.vel_to_cam_Tr[seq_index]
        lidar1 = self.vel2cam(lidar1, Tr)
        lidar2 = self.vel2cam(lidar2, Tr)
        gt_path = os.path.join(self.gt_root, f'{self.seqs_list[seq_index]:02d}_diff.npy')
        T_gt = self.read_diff_gt(gt_path, fn2)  # 这里得到的是两帧相机之间的变换，求逆才是两帧点云之间的变换
        T_gt_lidar = np.linalg.inv(T_gt)
        data_dict['lidar1'] = lidar1
        data_dict['lidar2'] = lidar2
        data_dict['T_gt_lidar'] = T_gt_lidar

        # 对点云进行数据预处理
        data_dict = self.apply_pre_processes(data_dict)

        # 获取四元数真值
        T_gt_lidar = data_dict['T_gt_lidar']
        R_gt = T_gt_lidar[:3, :3]
        t_gt = T_gt_lidar[:3, 3]
        z_gt, y_gt, x_gt = mat2euler(M=R_gt)
        q_gt = euler2quat(z=z_gt, y=y_gt, x=x_gt)
        t_gt = t_gt.astype(np.float32)  # (3,)
        q_gt = q_gt.astype(np.float32)  # (4,)

        lidar1 = data_dict['lidar1'].astype(np.float32)
        lidar2 = data_dict['lidar2'].astype(np.float32)
        lidar1_norm = lidar1
        lidar2_norm = lidar2

        return {'lidar1': lidar1, 'lidar2': lidar2, 'norm1': lidar1_norm, 'norm2': lidar2_norm,
                'q_gt': q_gt, 't_gt': t_gt}

    def get_seqs_len(self):
        seqs_len = []
        for seq_index in self.seqs_list:
            lidar_path = os.path.join(self.dataset_root, f'{seq_index:02d}', 'velodyne')
            seq_len = len(os.listdir(lidar_path))
            seqs_len.append(seq_len)
        return seqs_len

    def read_diff_gt(self, pose_path: str, index: int) -> np.ndarray:
        """
        注意返回的是index-th -> (index+1)-th相机位姿的变换,点云变换的话需要求对返回值求逆
        Args:
            pose_path: 路径
            index: 帧

        Returns: 从index-th帧相机到第(index+1)-th相机位姿的变换

        """
        poses = np.load(pose_path)
        pose = poses[index]
        filler = np.array([0.0, 0.0, 0.0, 1.0])
        pose = np.concatenate([pose, filler], axis=-1)
        pose = pose.reshape((4, 4))
        return pose

    def read_lidar(self, lidar_path: str):
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        lidar = lidar[:, :3]
        return lidar


    def vel2cam(self, lidar, Tr):
        lidar =  np.concatenate([lidar, np.ones((lidar.shape[0], 1))], axis=-1)
        lidar = lidar @ Tr.T
        lidar = lidar[:,:3]
        return lidar


