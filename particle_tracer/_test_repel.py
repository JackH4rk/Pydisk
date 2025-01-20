# -*- coding: utf-8 -*-
# @Time    : 2025/1/8 19:30
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : _test_repel.py
# @Software: PyCharm

import matplotlib
import numpy
import scipy.integrate

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import common

np = numpy
import scipy.constants as C
from _logging import logger

rmacro = 1e-9


def E(z, z_source, q):
    return q / (4 * numpy.pi * C.epsilon_0 * (numpy.abs(z - z_source + rmacro)) ** 3) * (z - z_source)


# def E(z,z_source,q):
#     return -1e200 * numpy.ones(z.shape)
def update(t, y, qi):
    zs_and_pzs = y.reshape((-1, 2))
    zs = zs_and_pzs[:, 0]
    pzs = zs_and_pzs[:, 1]
    d_zs = common.p_to_v(pzs, mass)
    d_pzs = qi * numpy.sum([E(zs, z_source, qi) for z_source in zs], axis=0)
    if d_pzs[0] >= 0:
        logger.info("Error, t = %.5e"%t)
        # _zs_for_plot =zs#numpy.array([ *numpy.linspace(zs.min()-10e-3,zs.min(),100),*zs,*numpy.linspace(zs.max(),zs.max() + 10e-3,100)])
        #
        # plt.plot(_zs_for_plot,  qi * numpy.sum([E(_zs_for_plot, z_source ,qi) for z_source in zs ],axis= 0), label = "%.2e"%t)
        # plt.scatter(zs, d_pzs)
        # plt.scatter(zs[0], d_pzs[0])
        # plt.legend()
        pass
    # has_inf_or_nan = numpy.isinf(d_pzs) | numpy.isnan(d_pzs)
    # if numpy.any( has_inf_or_nan) :
    #     logger.warning(d_pzs [ has_inf_or_nan])
    # logger.info(t)
    # _zs_for_plot =zs#numpy.array([ *numpy.linspace(zs.min()-10e-3,zs.min(),100),*zs,*numpy.linspace(zs.max(),zs.max() + 10e-3,100)])
    #
    # plt.plot(_zs_for_plot,  qi * numpy.sum([E(_zs_for_plot, z_source ,qi) for z_source in zs ],axis= 0),'.-' ,label = "%.6e"%t)
    # # plt.scatter(zs, d_pzs)
    # # plt.scatter(zs[0], d_pzs[0])
    # plt.legend()

    return numpy.array((d_zs, d_pzs)).T.reshape((-1,))


if __name__ == '__main__':
    plt.ion()
    Q = 2e-7 * 50
    z_initial = numpy.linspace(-0.1, 0.1, 50)
    qi =Q / len(z_initial)
    plt.figure()
    plt.plot(z_initial, E(z_initial, 0., qi))
    plt.xlabel("z (mm)")
    plt.ylabel("Ez (V/m)")


    vz_initial = 1e8
    mass = C.m_e * qi / C.e#1.1029332247741005e-18
    ts = numpy.linspace(0, 1 / 3e9, 2000)
    # plt.figure()

    sol = scipy.integrate.solve_ivp(update, (ts[0], ts[-1]),
                                    y0=numpy.array(
                                        [z_initial, mass * vz_initial * numpy.ones(z_initial.shape)]).T.reshape((-1,)),
                                    t_eval=ts,
                                    args=(qi,),
                                    dense_output=True,
                                    method = 'Radau'
                                    # first_step = 1e-18
                                    # maxstep = 1e-12,
                                    )
    traj_data = sol.y.reshape((len(z_initial), 2, -1))
    plt.figure()
    for i in numpy.array(list(range(len(z_initial))))[::]:  # [numpy.random.rand(len(z_i)) > 0.95]:
        plt.plot(
            sol.t,
            traj_data[i, 0, :],
        )

    plt.xlabel("time / s")
    plt.ylabel("z / m")

