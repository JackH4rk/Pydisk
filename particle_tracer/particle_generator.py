# -*- coding: utf-8 -*-
# @Time    : 2025/1/9 19:59
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : particle_generator.py
# @Software: PyCharm
import typing

import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy

import common
import scipy.constants as C


class ParticlePositionGenerator:
    def __init__(self, get_density: typing.Callable[[numpy.ndarray], numpy.ndarray]):
        self.get_density = get_density

    def generate(self, N, zmin, zmax,
                 dz=None) -> numpy.ndarray:
        """

        :param N:
        :param dz: 粒子区间，粒子在每个区间内均匀分布
        :return:
        """
        if dz is None:
            dz = (zmax - zmin) / N
        zs_ = numpy.arange(zmin, zmax, dz)
        density = self.get_density(zs_)
        # plt.figure()
        # plt.plot(zs_, density)
        sum_ = numpy.sum(density)

        i = 0
        last_i = 0
        N_generated = 0
        generated_zs = []

        while i < len(density): #N_generated< N:#
            p = numpy.sum(density[:i+1]) / sum_
            Np = int(N * p)
            if (Np - N_generated >= 1) and i >last_i:
                interval_lefts = numpy.array([*numpy.linspace(zs_[last_i],zs_[i],Np - N_generated ),zs_[i]])
                generated_zs.append([(interval_lefts[ii] + interval_lefts[ii+1])/2 for ii,_ in enumerate(interval_lefts[:-1])])
                # generated_zs.append((zs_[i] + zs_[last_i])/2 + (zs_[i] - zs_[last_i]) * numpy.linspace(-0.5, 0.5, Np - N_generated))
                last_i = i
                N_generated = Np
            i += 1
        return numpy.hstack(generated_zs)


if __name__ == '__main__':
    plt.ion()
    Ek = 50e3
    v = common.Ek_to_beta(Ek) * C.c
    f = 3e9
    L = v / f


    def modulated_density(z):
        return 1 + numpy.sin(2 * numpy.pi * z / L)


    ppg = ParticlePositionGenerator(modulated_density)
    zs = ppg.generate(1000, 0, 1 * L, #0.002*L
                      )
    plt.figure()
    plt.plot(zs, numpy.zeros(zs.shape), ".")

    plt.figure()
    plt.hist(zs, bins=200,density=True)
    _zs = numpy.linspace(zs.min(), zs.max(), len(zs))
    plt.plot(_zs,ppg.get_density(_zs) / (ppg.get_density(_zs).sum()*( _zs[1]-_zs[0]) ),)

