# -*- coding: utf-8 -*-
# @Time    : 2025/1/7 14:02
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : base.py
# @Software: PyCharm

import matplotlib

matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import numpy


class ResonantCav:
    def __init__(self, R_over_Q, Q0, Qe, omega_cav, ):
        self.R_over_Q = R_over_Q
        self.omega_cav = omega_cav
        self.Q0 = Q0
        self.Qe = Qe
        self.Q = 1 / (1 / Q0 + 1 / Qe)

    def calculate_Z_cav(self, omega):
        return self.R_over_Q * self.Q / (1 + 1j * self.Q * (omega / self.omega_cav - self.omega_cav / omega))

    def calculate_Gamma(self, omega):
        """
        计算反射系数
        :return:
        """
        beta = self.Q0 / self.Qe
        Y = self.Q0 * (omega / self.omega_cav - self.omega_cav / omega)

        return (beta - 1 - 1j * Y) / (beta + 1 + 1j * Y)

    def calculate_V_using_input_power(self, Pin, omega):
        R = self.R_over_Q * self.Q0
        Gamma = self.calculate_Gamma(omega)
        V = (2 * Pin * R * (1 - numpy.abs(Gamma) ** 2)) ** 0.5  # 与AJDISK计算完全一致
        return V

    def calculate_V(self, omega, I_ind_1):
        return self.calculate_Z_cav(omega, ) * I_ind_1


# class EzGeneratorBase(abc.ABC):
#     @abc.abstractmethod
#     def calculate_Ez(self, z, *args, **kwargs):
#         return 0.


if __name__ == '__main__':
    cav2 = ResonantCav(58.2, 2000, 175, 2 * numpy.pi * 2860e6)
    plt.figure()
    fs = numpy.linspace(2800e6, 3000e6, 1000)
    plt.plot(fs / 1e6, numpy.abs(cav2.calculate_Gamma(2 * numpy.pi * fs)))
    plt.axvline(2856, ls=':')
    plt.xlabel('frequency / MHz')
    plt.ylabel(r"reflection coefficient $\Gamma$")

    plt.figure()
    plt.plot(fs / 1e6, cav2.calculate_V_using_input_power(325, 2 * numpy.pi * fs))
    plt.axvline(2856, ls=':')
    plt.xlabel('frequency / MHz')
    plt.ylabel("V_cav (V)")
