# -*- coding: utf-8 -*-
# @Time    : 2024/11/24 20:48
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : disk.py
# @Software: PyCharm
import typing
import scipy.constants as C
import numpy


class RingsInfo:
    def __init__(self, qi, rout, rin, ids:typing.List[int], mass_kg = None):
        """
        不随粒子追踪过程变化的带电圆环集合的基本信息
        :param rin:
        :param rout:
        :param qi: 每个带电圆环的电荷量
        :param ids: 这些圆环的id
        """
        self.qi = qi
        self.rout = rout
        self.rin = rin
        self.ids = ids
        self.mass_kg = C.m_e * numpy.abs(qi / C.e) if mass_kg is None else mass_kg
class Rings:
    def __init__(self,info:RingsInfo):
        self.infos = info
        """
        trajdata: shape (Ntime, Nrings, 2), 其中最后一维的2个元素分别为 (z, vz)
        """
        self.trajdata :numpy.ndarray= None

