# -*- coding: utf-8 -*-
# @Time    : 2024/11/24 20:51
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : cavity.py
# @Software: PyCharm
import matplotlib

matplotlib.use('tkagg')

import numpy
import common
import scipy.constants as C

import field_generator.base as fb


class GaussProfileCavityField(  # fb.EzGeneratorBase,
    fb.ResonantCav):
    """
    符号定义：2014 Developing Sheet Beam Klystron Simulation Capability in AJDISK
    """

    def __init__(self, k, z0, R_over_Q, Q0, Qe, omega_cav, ):
        fb.ResonantCav.__init__(self, R_over_Q, Q0, Qe, omega_cav)
        # fb.EzGeneratorBase.__init__()

        self.z0 = z0
        self.k = k
        self.eq_length = 3 * (1 / 2 ** 0.5 / k)  # 3 sigma

    @staticmethod
    def calculate_k(beta_e, M):
        return 1 / 2 * beta_e / (-numpy.log(M))

    def f(self, z):
        return self.k / numpy.pi ** 0.5 * numpy.exp(-self.k ** 2 * (z - self.z0) ** 2)

    def calculate_Ez(self, z, Vcav, omega, t, *args, **kwargs):
        return Vcav * self.f(z) * numpy.exp(1j * omega * t)


if __name__ == '__main__':
    Ek_eV = 50e3
    beta_0 = common.Ek_to_beta(Ek_eV)
    f = 3e9
    L = C.c / f / 2

    cav1 = GaussProfileCavityField(1 / (L / 3), 10e-3, 50e6 / 1e4, 1e4, 1e10, 2.99e9)
