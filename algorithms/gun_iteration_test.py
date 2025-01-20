import numpy
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import charged
from field_generator.space_charge import RingSpaceChargeFieldGenerator
import common

def build_E_SC_interpolator(z_0_list, gamma,
                            # traj: numpy.ndarray, rings_info: charged.RingsInfo,
                            scg: RingSpaceChargeFieldGenerator) -> LinearNDInterpolator:
    """
    根据给定轨迹，构建空间电荷场的插值器
    :param traj: 参考algorithm中生成的traj_data的结构，即shape (N_particles, 2, N_time)，其中元素个数为2的那一维的两个元素分别是z, pz。
    :param rings_info: 电荷环自身数据
    :param scg: RingSpaceChargeFieldGenerator实例，用于计算空间电荷场
    :return: 空间电荷场的插值器intp，
             该插值器的调用方式：intp(t: float, z: numpy.ndarray) -> numpy.ndarray，即，输入时间t和位置z，返回该处、该位置空间电荷场的大小。
    """
    E_sc = np.sum(np.nan_to_num([scg.get_Es_in_lab_frame(z, z_0_list, gamma) for z in z_0_list]), axis=0)
    # E_sc = scg.get_Es_in_lab_frame(z_lab_frame, z0_lab_frame, gamma)
    # E_sc_cal = np.sum(np.nan_to_num([E_sc(z, z_0_list, gamma) for z in z_0_list]), axis=0)
    return E_sc

if __name__ == '__main__':
    pass
