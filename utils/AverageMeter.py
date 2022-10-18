# -*- coding:UTF-8 -*-

# author:Zhiheng Feng
# datetime:2022/2/25 16:37
# software: PyCharm

"""
文件说明：
    
"""
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum / self.count)