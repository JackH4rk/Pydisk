# -*- coding: utf-8 -*-
# @Time    : 2024/11/24 20:51
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : space_charge.py
# @Software: PyCharm
# 电场函数 (空间电荷场)
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import numpy
import scipy.constants as C
import scipy.interpolate
from scipy.special import j1, jn_zeros

import common

np = numpy
N_zeros = 100
mu_0_ps = jn_zeros(0, N_zeros)  # np.matrix(jn_zeros(0, N_zeros).reshape((-1, 1)))


class RingSpaceChargeFieldGenerator:
    def __init__(self, Q, a, b):
        """
        :param Q: 带电圆环电荷量
        :param a: 漂移管半径
        :param b: 束流半径
        """
        self.a = a
        self.b = b
        self.Q = Q

        z1_1 = 2e-3
        z1 = 10e-3
        z2 = 100e-3
        N1_1 = 2000
        N1 = 1000
        N2 = 100
        zs_ = numpy.array([*numpy.linspace(0, z1_1, N1_1, ),
                           *numpy.linspace(z1_1, z1, N1, ),
                           *numpy.linspace(z1, z2, N2, ), ])
        self.zs_ = numpy.hstack([-zs_[::-1], zs_])

        self.Es_data_ = self.__Es(self.zs_, Q, a, b)
        self.Es_interpolator = scipy.interpolate.interp1d(self.zs_, self.Es_data_, bounds_error=False, fill_value=0.0)

    def get_Es(self, z, z0, ):
        return self.Es_interpolator(z - z0, )

    @staticmethod
    def __Es(z: numpy.ndarray, Q, a, b):
        """
        Ref: AJDISK help
        To reemphasize, b is the beam radius, a is the drift tube radius, Q is the disk charge, z0 is the location of the disk, and z is the variable associated with location.

        :param z:
        :param Q:
        :param a:
        :param b:
        :return:
        """
        arr = numpy.array(
            [np.exp(-mu_0_p * np.abs(z) / a) * (((2 / mu_0_p * j1(mu_0_p * b / a) / j1(mu_0_p)) ** 2) * np.sign(z)) for
             mu_0_p in mu_0_ps])
        return Q / (2 * np.pi * C.epsilon_0 * b ** 2) * arr.sum(axis=0)

    def get_Es_in_lab_frame(self, z_lab_frame, z0_lab_frame, gamma):
        return self.Es_interpolator(gamma * (z_lab_frame - z0_lab_frame), )
        # return self.__Es(gamma * (z_lab_frame - z0_lab_frame),self. Q,self. a,self. b)


if __name__ == '__main__':
    plt.ion()
    Q = 1.9398610262781608e-08
    b = 3e-3
    a = 4e-3
    z0 = 10e-3 * 0
    Dz = 200e-3
    scg = RingSpaceChargeFieldGenerator(Q, a, b)
    zs = scg.zs_  # numpy.linspace(z0- Dz / 2,z0+Dz /2 , 1000)

    plt.figure()
    plt.plot(zs, zs, '.')
    plt.figure()
    plt.plot(zs / 1e-3, scg.get_Es(zs, z0,),label ="in rest frame" )
    plt.plot(zs / 1e-3, scg.get_Es_in_lab_frame(zs, z0,common.Ek_to_gamma(50e3)),label = 'in lab frame')
    plt.legend()