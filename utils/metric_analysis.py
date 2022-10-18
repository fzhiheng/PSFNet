# -- coding: utf-8 --

import pandas as pd


# @Time : 2022/10/18 15:30
# @Author : Zhiheng Feng
# @File : metric_analysis.py
# @Software : PyCharm


def odo_metric_analysis(metric: pd.DataFrame, key_list: list) -> pd.DataFrame:
    """
    对得到的里程计在多个序列上的指标进行分析，找出几个中平移和旋转最小的时候的epoch
    Args:
        metric: 需要进行分析的表
        key_list: metric中指标的名称

    Returns: 在输入的表格后拼接上分析数据返回

    """
    metric_dict = {}
    for metric_name in key_list:
        for column in metric.columns:
            if metric_name in column:
                metric_dict.setdefault(metric_name, []).append(column)
            else:
                continue
    for key, value in metric_dict.items():
        df_tmp = metric[value]
        series_tmp = df_tmp.mean(axis='columns')
        metric[f'mean_{key}'] = series_tmp
        metric[f'min_{key}_epoch'] = pd.Series([series_tmp.idxmin()], index=[metric.index[0]])






