# -*- coding: utf-8 -*-
# @Time    : 2024/11/24 21:08
# @Author  : Z.J. Zhang
# @Email   : zijingzhang@mail.ustc.edu.cn
# @File    : algorithm.py
# @Software: PyCharm
import time
import matplotlib
import numpy
import scipy.integrate
from _logging import logger

import particle_tracer.particle_generator
from algorithms.gun_iteration_test import build_E_SC_interpolator

matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import common

np = numpy
import scipy.constants as C
import charged
import klystron
from field_generator.cavity import GaussProfileCavityField
import field_generator.space_charge as sc
class DiskAlgorithm:
    def __init__(self, f0, charged_rings: charged.Rings, klystron_info: klystron.KlystronInfo):
        pass

cnt = 0
def update(t, y, Vcav: complex, omega, rings_info: charged.RingsInfo, cav: GaussProfileCavityField,
           space_charge_generator: sc.RingSpaceChargeFieldGenerator,
           zmin, zmax,
           ):
    """

    :param t:
    :param y: (z0, pz0, z1, pz1, ...)
    :param Vcav:
    :param omega:
    :param rings_info:
    :param cav:
    :param space_charge_generator:
    :return:
    """
    global cnt
    cnt += 1

    #总迭代次数
    print(f'总第{cnt}次迭代')

    # print(zs_and_pzs)

    # γ的单时间点修正
    # print('zi＆pzi', zs_and_pzs)
    # print('Zmin-Zmax间盘片的γ', gammas)
    # print('本时刻γ均值', gamma)

    zs_and_pzs = y.reshape((-1, 2))
    zs = zs_and_pzs[:, 0]
    pzs = zs_and_pzs[:, 1]
    _filter = (zs > zmin) & (zs < zmax) # 在此范围之外，既不受任何力，也不产生任何力（即不存在对其他粒子的空间电荷力）
    gammas = common.gammabeta_to_gamma(pzs[_filter] / (rings_info.mass_kg * C.c))
    # if not numpy.any(_filter):
    gamma = np.mean(gammas)
    gamma = 1 if np.isnan(gamma) else gamma



    F = numpy.piecewise(
        zs, [_filter],
        [lambda z0:
            rings_info.qi * (cav.calculate_Ez(z0, Vcav, omega, t).real
                             + build_E_SC_interpolator(z0, gamma, space_charge_generator)
            # + np.sum(np.nan_to_num([space_charge_generator.get_Es_in_lab_frame(z, z0, gamma) for z in z0]), axis=0)
            # + sc_intp(z,gamma))
            # + numpy.sum([space_charge_generator.get_Es_in_lab_frame(z, z0, gammas[_filter][i]) for i, z0 in
            #               enumerate(zs[_filter])], axis=0)  # 应调用interpolator
        ),
         0])
    # logger.info("%.2e"%t)
    d_zs = common.p_to_v(pzs, rings_info.mass_kg)
    d_pzs = F
    # plt.figure()
    # plt.plot(zs,d_zs)
    # plt.figure()
    # plt.plot(zs, d_pzs,'.-')

    res = numpy.array((d_zs, d_pzs)).T.reshape((-1,))
    return res


def simulation():
    pass


def min_and_max(arr):
    return min(arr), max(arr)


if __name__ == '__main__':

    N_cnt = 0
    plt.ion()
    f_0 = 3.e9
    ts = numpy.linspace(0 / f_0, 10.0 / f_0, 1000)
    v_i = common.Ek_to_beta(50e3) * C.c  # 电子速度

    f_cav = 3.05e9

    cav = GaussProfileCavityField(GaussProfileCavityField.calculate_k(2 * numpy.pi / (v_i / f_0), M=0.7),
                                  20e-3, 80,
                                  2000, 1e10, 2 * numpy.pi * f_cav)
    cav_eq_length = cav.eq_length

    simulation_zone_zmin = cav.z0 - cav_eq_length
    simulation_zone_zmax = cav.z0 + cav_eq_length

    # z_i_ = numpy.linspace(simulation_zone_zmin
    #                       - (ts[-1] - ts[0]) * v_i  # 预填充整个模拟期间会进入模拟区域的粒子
    #                       ,
    #                       simulation_zone_zmax, int(100)
    #                       )
    N_disk = 50
    L_initial_z_distribution = 1 / f_0 * v_i
    z_i = particle_tracer.particle_generator.ParticlePositionGenerator(
        lambda z: 0.05 * numpy.sin(2 * numpy.pi / (v_i / f_0) * z) + 1, ).generate(
        N_disk,
        simulation_zone_zmin
        - L_initial_z_distribution,  # 预填充整个模拟期间会进入模拟区域的粒子
        simulation_zone_zmin,
        L_initial_z_distribution / 2000
    )
    # plt.hist(numpy.random.rand(len(z_i)))

    omega = 2 * numpy.pi * f_0
    I_avg = 120  # 0
    # 初始位置
    # z_i = z_i_   [numpy.sin(2 * numpy.pi / (v_i / f_0) * z_i_) + 1 > 2 * (numpy.random.rand(len(z_i_)))]

    Q_tot = I_avg * (z_i[-1] - z_i[0]) / v_i  # 总电荷量
    qi = Q_tot / len(z_i)  # 单个盘片的电荷量
    rings_info = charged.RingsInfo(qi, 3e-3, 0, list(range(len(z_i))))

    pz_i = common.beta_to_betagamma(v_i / C.c) * rings_info.mass_kg * C.c

    # plt.figure()
    # plt.hist(z_i, bins=10)

    plt.figure()
    plt.plot(z_i, numpy.zeros(z_i.shape), '.', alpha=1)

    # n_L, z_bins = numpy.histogram(z_i, bins=int((z_i[-1] - z_i[0]) / (v_i / f_0) * 50),
    #                               density=True)
    # z_bins = z_bins[:-1]
    plt.figure()
    __zs_for_plot = numpy.linspace(z_i.min(), z_i.max(), 100)
    plt.plot(__zs_for_plot, numpy.sum(
        [rings_info.qi * common.Gaussian(z_i, z0, L_initial_z_distribution / 50) * v_i for z0 in __zs_for_plot],
        axis=1))
    plt.ylabel("I_beam (A)")

    r_drift = 4e-3

    V_cav_history = [0e9 + 0j]
    Delta_Vcav_history = []
    sc_generator = sc.RingSpaceChargeFieldGenerator(qi, r_drift, rings_info.rout)

    # plt.figure()
    # plt.plot(z_i_, cav.calculate_Ez(z_i_, V_cav_history[-1], omega, 0))
    iteration = 0
    traj_data = None
    sol = None
    traj_data_for_calculate_I_ind = None
    I_ind = None
    I_ind_1 = None
    Vcav_new = V_cav_history[-1]


    # aaaaaaa
    def get_residual(Vcav_real_and_imag: numpy.ndarray):  #  添加一个参数E_sc_interpolator
        global traj_data, sol, traj_data_for_calculate_I_ind, I_ind, I_ind_1, Vcav_new
        Vcav = Vcav_real_and_imag[0] + 1j * Vcav_real_and_imag[1]

        # Vcav = V_cav_history[-1]
        V_cav_history.append(Vcav)
        logger.info("|Vcav| = %.2e V" % (  # iteration,
            numpy.abs(Vcav)))
        sol = scipy.integrate.solve_ivp(update, (ts[0], ts[-1]),
                                        y0=numpy.array([z_i, pz_i * numpy.ones(z_i.shape)]).T.reshape((-1,)),
                                        t_eval=ts,
                                        args=(Vcav, omega, rings_info, cav, sc_generator, simulation_zone_zmin,
                                              simulation_zone_zmax),
                                        # first_step = 1e-18
                                        #  method='Radau'

                                        )
        traj_data = sol.y.reshape((len(z_i), 2, -1))
        traj_data_interpolator = scipy.interpolate.interp1d(sol.t, traj_data, bounds_error=False, fill_value=numpy.nan)

        traj_data_for_calculate_I_ind = numpy.zeros((3 * traj_data.shape[0], *traj_data.shape[1:]))
        traj_data_for_calculate_I_ind[:len(traj_data)] = traj_data
        traj_data_for_calculate_I_ind[len(traj_data):2 * len(traj_data)] = traj_data_interpolator(sol.t + 1 / f_0)
        traj_data_for_calculate_I_ind[2 * len(traj_data):] = traj_data_interpolator(sol.t - 1 / f_0)

        # traj_data_for_calculate_I_ind = numpy.zeros((5 * traj_data.shape[0], *traj_data.shape[1:]))
        # traj_data_for_calculate_I_ind[:len(traj_data)] = traj_data_interpolator(sol.t + 2 / f_0)
        # traj_data_for_calculate_I_ind[len(traj_data):2 * len(traj_data)] = traj_data_interpolator(sol.t + 1 / f_0)
        # traj_data_for_calculate_I_ind[2 * len(traj_data):3 * len(traj_data)] = traj_data
        # traj_data_for_calculate_I_ind[3 * len(traj_data):4 * len(traj_data)] = traj_data_interpolator(sol.t - 1 / f_0)
        # traj_data_for_calculate_I_ind[4 * len(traj_data):] = traj_data_interpolator(sol.t - 2 / f_0)

        # traj_data_for_calculate_I_ind[:len(traj_data)] = traj_data_interpolator(sol.t + 1 / f_0)
        # traj_data_for_calculate_I_ind[len(traj_data):2 * len(traj_data)] = traj_data
        # traj_data_for_calculate_I_ind[2 * len(traj_data):] = traj_data_interpolator(sol.t - 1 / f_0)

        I_ind = numpy.nansum(
            [rings_info.qi * cav.f(traj_data_for_calculate_I_ind[i, 0, :]) * common.p_to_v(
                traj_data_for_calculate_I_ind[i, 1, :], rings_info.mass_kg) for i in
             range(len(traj_data_for_calculate_I_ind))], axis=0)
        # plt.figure()
        # plt.plot(sol.t / (1/f_0), I_ind)
        # plt.xlim(0,1)

        I_ind_interpolator = scipy.interpolate.interp1d(sol.t, I_ind)
        dt = sol.t[1] - sol.t[0]
        # ts_for_FourierSeriesAnalysis = numpy.arange((ts[0] + ts[-1]) / 2 - 0.5 / f_0, (ts[0] + ts[-1]) / 2 + 0.5 / f_0 ,
        #                                             dt)
        ts_for_FourierSeriesAnalysis = numpy.arange(0, 1 / f_0,
                                                    dt)
        _a1, _b1 = [(I_ind_interpolator(ts_for_FourierSeriesAnalysis) * numpy.cos(
            2 * numpy.pi * f_0 * ts_for_FourierSeriesAnalysis) * dt).sum() * 2 / (
                            ts_for_FourierSeriesAnalysis[-1] - ts_for_FourierSeriesAnalysis[0]),
                    (I_ind_interpolator(ts_for_FourierSeriesAnalysis) * numpy.sin(
                        2 * numpy.pi * f_0 * ts_for_FourierSeriesAnalysis) * dt).sum() * 2 / (
                            ts_for_FourierSeriesAnalysis[-1] - ts_for_FourierSeriesAnalysis[0])]
        I_ind_1 = _a1 + 1j * _b1
        logger.info("I_ind_1 / I_avg = %.2f" % (numpy.abs(I_ind_1) / I_avg))

        Vcav_new = I_ind_1 * cav.calculate_Z_cav(2 * numpy.pi * f_0)

        # V_cav_history.append(Vcav)

        # iteration += 1
        Delta_V = numpy.abs(Vcav_new - Vcav)
        Delta_Vcav_history.append(Delta_V)
        logger.info("Delta_V = %.5e V" % Delta_V)
        # convergence_criterion_1 = Delta_V - numpy.abs(V_cav_history[-2]) * 1e-6
        # logger.info("(numpy.abs(Vcav - V_cav_history[-2])) = %.2e\n numpy.abs(V_cav_history[-2]) * 1e-6 = %.2e" % (
        #     Delta_V,
        #     numpy.abs(V_cav_history[-2]) * 1e-6))
        # break

        # 每轮枪迭代法后结果
        # print('zi＆pzi', sol.y)

        return Delta_V ** 2


    _debug_t = time.time()


    def real_and_imag(c: complex):
        return c.real, c.imag
    #  初始化可置none

    for i in range(50):
        N_cnt += 1
        print(f'总第{N_cnt}轮迭代')
        get_residual(numpy.array(real_and_imag(V_cav_history[-1] + Vcav_new)) / 2)
    # min_res = minimize(get_residual,[Vcav_new.real,Vcav_new.imag ],)
    logger.info("Time cost = %.5f s" % (time.time() - _debug_t))



    plt.figure()
    plt.plot(list(range(len(V_cav_history))), numpy.abs(V_cav_history), label="V_cav_history")
    # plt.title("V_cav_history")

    # plt.figure()
    plt.plot(list(range(len(Delta_Vcav_history))), Delta_Vcav_history, label="DV_cav_history")
    # plt.title("DV_cav_history")
    plt.legend()
    # aaaaaaaaaaaa
    #
    #
    # while True:
    #     Vcav = V_cav_history[-1]
    #     logger.info("Iteration = %d, |Vcav| = %.2e V" % (iteration, numpy.abs(Vcav)))
    #     sol = scipy.integrate.solve_ivp(update, (ts[0], ts[-1]),
    #                                     y0=numpy.array([z_i, pz_i * numpy.ones(z_i.shape)]).T.reshape((-1,)),
    #                                     t_eval=ts,
    #                                     args=(Vcav, omega, rings_info, cav, sc_generator, simulation_zone_zmin,
    #                                           simulation_zone_zmax),
    #                                     # first_step = 1e-18
    #                                     # method='Radau'
    #
    #                                     )
    #     traj_data = sol.y.reshape((len(z_i), 2, -1))
    #     traj_data_interpolator = scipy.interpolate.interp1d(sol.t, traj_data,bounds_error=False, fill_value=numpy.nan)
    #
    #     traj_data_for_calculate_I_ind = numpy.zeros((2*traj_data.shape[0],*traj_data.shape[1:]))
    #     traj_data_for_calculate_I_ind[:len(traj_data)] = traj_data
    #     traj_data_for_calculate_I_ind[len(traj_data):2 * len(traj_data)] = traj_data_interpolator(sol.t + 1/f_0)
    #     # traj_data_for_calculate_I_ind[2 * len(traj_data): ] = traj_data_interpolator(sol.t - 1/f_0)
    #
    #
    #
    #     I_ind = numpy.sum(
    #         [rings_info.qi * cav.f(traj_data_for_calculate_I_ind[i, 0, :]) * common.p_to_v(traj_data_for_calculate_I_ind[i, 1, :], rings_info.mass_kg) for i in
    #          range(len(traj_data_for_calculate_I_ind))], axis=0)
    #
    #     I_ind_interpolator = scipy.interpolate.interp1d(sol.t, I_ind)
    #     dt = sol.t[1] - sol.t[0]
    #     # ts_for_FourierSeriesAnalysis = numpy.arange((ts[0] + ts[-1]) / 2 - 0.5 / f_0, (ts[0] + ts[-1]) / 2 + 0.5 / f_0 ,
    #     #                                             dt)
    #     ts_for_FourierSeriesAnalysis = numpy.arange(0, 1/ f_0,
    #                                                 dt)
    #     _a1, _b1 = [(I_ind_interpolator(ts_for_FourierSeriesAnalysis) * numpy.sin(
    #         2 * numpy.pi * f_0 * ts_for_FourierSeriesAnalysis) * dt).sum() * 2 / (
    #                         ts_for_FourierSeriesAnalysis[-1] - ts_for_FourierSeriesAnalysis[0]),
    #                 (I_ind_interpolator(ts_for_FourierSeriesAnalysis) * numpy.cos(
    #                     2 * numpy.pi * f_0 * ts_for_FourierSeriesAnalysis) * dt).sum() * 2 / (
    #                         ts_for_FourierSeriesAnalysis[-1] - ts_for_FourierSeriesAnalysis[0])]
    #     I_ind_1 = _a1 + 1j * _b1
    #     logger.info("I_ind_1 / I_avg = %.2f"%(numpy.abs(I_ind_1) / I_avg))
    #     Vcav = I_ind_1 * cav.calculate_Z_cav(2 * numpy.pi * f_0)
    #     V_cav_history.append(Vcav)
    #
    #     iteration += 1
    #     Delta_V = numpy.abs(Vcav - V_cav_history[-2])
    #     convergence_criterion_1 = Delta_V - numpy.abs(V_cav_history[-2]) * 1e-6
    #     logger.info("(numpy.abs(Vcav - V_cav_history[-2])) = %.2e\n numpy.abs(V_cav_history[-2]) * 1e-6 = %.2e" % (
    #         Delta_V,
    #         numpy.abs(V_cav_history[-2]) * 1e-6))
    #     # break
    #     if (convergence_criterion_1 < 0) or (iteration > 50):
    #         if convergence_criterion_1 < 0:
    #             logger.info("Converged!")
    #         break
    # logger.info("Time cost = %.5f s"%(time.time() - _debug_t))
    # plt.figure()
    # plt.plot(list(range(len(V_cav_history))), numpy.abs(V_cav_history))
    # plt.title("V_cav_history")

    # ax2: plt.Axes = plt.twiny(plt.gca())
    # ax2.plot(cav.calculate_Ez(z_i_, 1, omega, 0), z_i_, )
    Ts, Zs = numpy.meshgrid(ts, numpy.array([*numpy.linspace(traj_data[:, 0, :].min(), cav.z0 - cav_eq_length, 20, ),
                                             *numpy.linspace(cav.z0 - cav_eq_length, cav.z0 + cav_eq_length,
                                                             20, ),
                                             *numpy.linspace(cav.z0 + cav_eq_length, traj_data[:, 0, :].max(),
                                                             20, ),
                                             ],

                                            ))
    plt.figure()
    for i in numpy.array(list(range(len(z_i))))[::]:  # [numpy.random.rand(len(z_i)) > 0.95]:
        plt.plot(
            sol.t,
            traj_data[i, 0, :]
        )
    cf = plt.contourf(Ts, Zs, cav.calculate_Ez(Zs, V_cav_history[-2], omega, Ts).real, zorder=-1,
                      cmap=plt.get_cmap('jet'), levels=30)
    plt.colorbar(cf)
    plt.axhline(simulation_zone_zmin, ls=':')
    plt.axhline(simulation_zone_zmax, ls=':')
    plt.xlabel("time / s")
    plt.ylabel("z / m")

    plt.figure()
    for i in numpy.array(list(range(len(traj_data_for_calculate_I_ind))))[::]:  # [numpy.random.rand(len(z_i)) > 0.95]:
        plt.plot(
            sol.t,
            traj_data_for_calculate_I_ind[i, 0, :]
        )
    cf = plt.contourf(Ts, Zs, cav.calculate_Ez(Zs, V_cav_history[-2], omega, Ts).real, zorder=-1,
                      cmap=plt.get_cmap('jet'), levels=30)
    plt.colorbar(cf)
    plt.axhline(simulation_zone_zmin, ls=':')
    plt.axhline(simulation_zone_zmax, ls=':')
    plt.xlabel("time / s")
    plt.ylabel("z / m")

    plt.figure()
    plt.plot(sol.t / (1 / f_0), I_ind, label="I_ind")
    plt.axhline(I_avg, ls=':', label="initial setting")
    plt.xlabel("time (1/f_0)")
    plt.ylabel("I_ind (A)")
    # plt.ylim(0, None)
    plt.xlim(0, 1)
    plt.legend()

    plt.figure()
    plt.hist(traj_data[:, 0, -1], bins=20)

    logger.info("初始平均能量%.2f keV" % (numpy.nanmean((common.gammabeta_to_gamma(
        traj_data_for_calculate_I_ind[:, 1, 0] / (rings_info.mass_kg * C.c)) - 1) * C.m_e * C.c ** 2) / C.eV / 1e3))
    logger.info("最终平均能量%.2f keV" % (numpy.nanmean((common.gammabeta_to_gamma(
        traj_data_for_calculate_I_ind[:, 1, -1] / (rings_info.mass_kg * C.c)) - 1) * C.m_e * C.c ** 2) / C.eV / 1e3))
