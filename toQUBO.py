import numpy as np
import pickle
from main import SBM


class QUBO:
    @staticmethod
    def params2qubo_v2(rsrp, capacity, rb, serving_list, ue_num, bs_num, spin=False):
        def vecmul(vec1, vec2, result=None):
            l1 = len(vec1)
            l2 = len(vec2)
            if result is None:
                result = np.zeros([l1, l2])
            for i in range(l1):
                if vec1[i] == 0:
                    continue
                result[i] += vec1[i] * vec2
            return result

        max_rbnum_ue = 200
        x_dim = ue_num * bs_num

        # throughput fo each ue, must under 200, 10*19+(1+2+4+2)
        digit_num = 4  # 4 for len of [1, 2, 4, 2]
        robinmax = (max_rbnum_ue // 10 - 1)
        robin10_dim = bs_num * robinmax  # t == 19
        robin2_dim = ue_num * bs_num * digit_num  # d == 1050*4

        # 1. constrain of numbers fo ue conneting to each bs are not greater than 128
        power128 = 7 + 1  # summation have to > 128
        r_dim = bs_num * power128

        # 2. constrain 2 has no slack variable

        # 3. constrain of demand-throughput >= 0
        s_dim = ue_num * bs_num * power128

        # 4. constrain of maximum number of bs is 273
        power256 = 8 + 1  # summation have to > 273
        y_dim = ue_num * bs_num * robinmax
        ybar_dim = ue_num * bs_num * digit_num
        u_dim = bs_num * power256

        # 5. CIO
        ##################################################
        ### index of e is definded j * j'(j' != j) * 6 ###
        ### 0, [1,2,3,4,5,6], [:]                      ###
        ### 1, [0,2,3,4,5,6], [:]                      ###
        ### 2, [0,1,3,4,5,6], [:]                      ###
        ### ...                                        ###
        ##################################################
        cio_range_dim = 6  # 6 for len of [1, 2, 4, 8, 16, 9]
        e_dim = bs_num * bs_num * cio_range_dim
        v_dim = ue_num * bs_num * power128

        # init jh matrix
        robin10_shift = x_dim
        robin2_shift = robin10_shift + robin10_dim
        r_shift = robin2_shift + robin2_dim
        s_shift = r_shift + r_dim
        y_shift = s_shift + s_dim
        ybar_shift = y_shift + y_dim
        u_shift = ybar_shift + ybar_dim
        e_shift = u_shift + u_dim
        v_shift = e_shift + e_dim
        h_dim = v_shift + v_dim
        h = np.zeros([h_dim + 1, h_dim + 1])

        # Hamiltonian
        h2d = np.zeros([h_dim + 1, h_dim + 1])
        for j in range(bs_num):
            for i in range(ue_num):
                rb_bar = int(rb[i, j] // 10)
                if rb_bar > 19:
                    rb_bar = 19
                for k in range(rb_bar):
                    h2d[robin10_shift + j * robinmax + k, i * bs_num + j] += -10 * 0.5 * 156 * capacity[
                        i, j] / 450 / 2
                    h2d[i * bs_num + j, robin10_shift + j * robinmax + k] += -10 * 0.5 * 156 * capacity[
                        i, j] / 450 / 2
                for k in range(digit_num):
                    if k == digit_num - 1:
                        h2d[robin2_shift + i * (bs_num * digit_num) + j * digit_num + k, i * bs_num + j] += \
                            -2 * 0.5 * 156 * capacity[i, j] / 450 / 2
                        h2d[i * bs_num + j, robin2_shift + i * (bs_num * digit_num) + j * digit_num + k] += \
                            -2 * 0.5 * 156 * capacity[i, j] / 450 / 2
                    else:
                        h2d[robin2_shift + i * (bs_num * digit_num) + j * digit_num + k, i * bs_num + j] += \
                            -2 ** k * 0.5 * 156 * capacity[i, j] / 450 / 2
                        h2d[i * bs_num + j, robin2_shift + i * (bs_num * digit_num) + j * digit_num + k] += \
                            -2 ** k * 0.5 * 156 * capacity[i, j] / 450 / 2

        # constrain 1
        c12d = np.zeros([h_dim + 1, h_dim + 1])
        for j in range(bs_num):
            c11d = np.zeros([h_dim + 1])
            for i in range(ue_num):
                c11d[i * bs_num + j] = 1
            for k in range(power128):
                c11d[r_shift + j * power128 + k] = 2 ** k
            c11d[-1] = -128
            c12d = vecmul(c11d, c11d, c12d)

        # constrain 2
        c22d = np.zeros([h_dim + 1, h_dim + 1])
        for i in range(ue_num):
            c21d = np.zeros([h_dim + 1])
            for j in range(bs_num):
                c21d[i * bs_num + j] = 1
            c21d[-1] = -1
            c22d = vecmul(c21d, c21d, c22d)

        # constrain 3
        c32d = np.zeros([h_dim + 1, h_dim + 1])
        for i in range(ue_num):
            for j in range(bs_num):
                c31d = np.zeros([h_dim + 1])

                # demand rb
                c31d[-1] = rb[i, j]

                # rb distrubution
                rb_bar = int(rb[i, j] // 10)
                if rb_bar > 19:
                    rb_bar = 19
                for k in range(rb_bar):
                    c31d[robin10_shift + j * robinmax + k] -= 10
                for k in range(digit_num):
                    if k == digit_num - 1:
                        c31d[robin2_shift + i *
                             (bs_num * digit_num) + j * digit_num + k] -= 2
                    else:
                        c31d[robin2_shift + i *
                             (bs_num * digit_num) + j * digit_num + k] -= 2 ** k

                # slack variable
                for k in range(power128):
                    c31d[s_shift + i * (bs_num * power128) +
                         j * (power128) + k] -= 2 ** k
                c32d = vecmul(c31d, c31d, c32d)

        # constrain 4
        c42d = np.zeros([h_dim + 1, h_dim + 1])
        for j in range(bs_num):
            c41d = np.zeros([h_dim + 1])

            # constrain : distru < 273
            c41d[-1] = -273

            # rb distrubution
            for i in range(ue_num):
                rb_bar = int(rb[i, j] // 10)
                if rb_bar > 19:
                    rb_bar = 19
                for k in range(rb_bar):
                    c41d[y_shift + i * (bs_num * robinmax) +
                         j * robinmax + k] += 10
                for k in range(digit_num):
                    if k == digit_num - 1:
                        c41d[ybar_shift + i *
                             (bs_num * digit_num) + j * digit_num + k] += 2
                    else:
                        c41d[ybar_shift + i *
                             (bs_num * digit_num) + j * digit_num + k] += 2 ** k

            # slack variable
            for k in range(power256):
                c41d[u_shift + j * power256 + k] += 2 ** k

            c42d = vecmul(c41d, c41d, c42d)

        p = 1
        for i in range(ue_num):
            for j in range(bs_num):
                for k in range(robinmax):
                    c42d[robin10_shift + j * robinmax +
                         k, i * bs_num + j] += p * 1
                    c42d[i * bs_num + j, robin10_shift +
                         j * robinmax + k] += p * 1
                    c42d[robin10_shift + j * robinmax + k, y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k] += p * -2
                    c42d[y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k, robin10_shift + j * robinmax + k] += p * -2
                    c42d[i * bs_num + j, y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k] += p * -2
                    c42d[y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k, i * bs_num + j] += p * -2
                    c42d[y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k, -1] += p * 3
                    c42d[-1, y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k] += p * 3

                for k in range(digit_num):
                    c42d[robin2_shift + j * robinmax +
                         k, i * bs_num + j] += p * 1
                    c42d[i * bs_num + j, robin2_shift +
                         j * robinmax + k] += p * 1
                    c42d[robin2_shift + j * robinmax + k, y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k] += p * -2
                    c42d[y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k, robin2_shift + j * robinmax + k] += p * -2
                    c42d[i * bs_num + j, y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k] += p * -2
                    c42d[y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k, i * bs_num + j] += p * -2
                    c42d[y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k, -1] += p * 3
                    c42d[-1, y_shift + i * (bs_num * robinmax) + j * (
                        robinmax) + k] += p * 3

        # constrain 5
        c52d = np.zeros([h_dim + 1, h_dim + 1])
        for i in range(ue_num):
            for j in range(bs_num):
                serving_idx = int(serving_list[i])
                if serving_idx == j:
                    continue
                c51d = np.zeros([h_dim + 1])

                c51d[i * bs_num + j] = rsrp[i, j] - rsrp[i, serving_idx] - 10
                for k in range(cio_range_dim):
                    if k == cio_range_dim - 1:
                        c51d[e_shift + serving_idx *
                             (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 9 / 2
                    else:
                        c51d[e_shift + serving_idx *
                             (bs_num * cio_range_dim) + j * cio_range_dim + k] -= 2 ** (k - 1)
                c51d[-1] += 20

                for k in range(power128):
                    if k == power128 - 1:
                        c51d[v_shift + i * (bs_num * power128) +
                             j * power128 + k] -= 185 / 2
                    else:
                        c51d[v_shift + i * (bs_num * power128) +
                             j * power128 + k] -= 2 ** (k - 1)
                c51d[-1] += 110
                c52d = vecmul(c51d, c51d, c52d)

        # constrain 6
        c62d = np.zeros([h_dim + 1, h_dim + 1])
        for j in range(bs_num):
            for k in range(robinmax - 1):
                c62d[robin10_shift + j * robinmax + k + 1, -1] += 1 / 2
                c62d[-1, robin10_shift + j * robinmax + k + 1] += 1 / 2
                c62d[robin10_shift + j * robinmax + k + 1,
                     robin10_shift + j * robinmax + k] -= 1 / 2
                c62d[robin10_shift + j * robinmax + k,
                     robin10_shift + j * robinmax + k + 1] -= 1 / 2

        result = h2d + c12d + c22d + c32d + c42d + c52d + c62d

        if spin:
            jh = np.zeros([h_dim+1, h_dim+1])
            for i in range(h_dim):
                for j in range(h_dim):
                    c2 = result[i, j]
                    jh[i, j] = c2
                    jh[i, -1] += -c2/2
                    jh[-1, i] += -c2 / 2
                    jh[j, -1] += -c2 / 2
                    jh[-1, j] += -c2 / 2
                    jh[-1, -1] += c2 / 4

                jh[i, -1] += result[i, -1]
                jh[-1, i] += result[-1, i]
                jh[-1, -1] += -(result[i, -1]+result[-1, i])/2
            result = jh

        return result, h2d, c12d, c22d, c32d, c42d, c52d, c62d

    @staticmethod
    def params2qubo(rsrp, sinr, rb, ue_num, bs_num):
        assert (ue_num, bs_num) == rsrp.shape, "rsrp must have same shape with ({}, {})," \
                                               " but given {}".format(
                                                   ue_num, bs_num, rsrp.shape)
        assert (ue_num, bs_num) == sinr.shape, "sinr must have same shape with ({}, {})," \
                                               " but given {}".format(
                                                   ue_num, bs_num, sinr.shape)
        assert (ue_num, bs_num) == rb.shape, "rb must have same shape with ({}, {})," \
                                             " but given {}".format(
                                                 ue_num, bs_num, rb.shape)

        bin_size = bs_num * ue_num
        Q = np.zeros([bin_size, bin_size])
        for j in range(bs_num):
            for i in range(ue_num):
                for k in range(ue_num):
                    Q[i + j * ue_num, k + j * ue_num] += rb[i, j] * rb[k, j]

        for k in range(bs_num):
            for l in range(ue_num):
                for j in range(bs_num):
                    for i in range(ue_num):
                        Q[i + j * ue_num, l + k * ue_num] += - \
                            1 * rb[i, j] * rb[l, k] / bs_num

        Q /= bs_num
        return Q

    @staticmethod
    def params2jh(rsrp, sinr, rb, ue_num, bs_num):
        assert (ue_num, bs_num) == rsrp.shape, "rsrp must have same shape with ({}, {})," \
                                               " but given {}".format(
                                                   ue_num, bs_num, rsrp.shape)
        assert (ue_num, bs_num) == sinr.shape, "sinr must have same shape with ({}, {})," \
                                               " but given {}".format(
                                                   ue_num, bs_num, sinr.shape)
        assert (ue_num, bs_num) == rb.shape, "rb must have same shape with ({}, {})," \
                                             " but given {}".format(
                                                 ue_num, bs_num, rb.shape)

        bin_size = bs_num * ue_num
        J = np.zeros([bin_size, bin_size])
        H = np.zeros([bin_size])
        C = 0
        for j in range(bs_num):
            for i in range(ue_num):
                for k in range(ue_num):
                    coe1 = rb[i, j] * rb[k, j]
                    J[i + j * ue_num, k + j * ue_num] += coe1
                    H[i + j * ue_num] += coe1 / 2
                    H[k + j * ue_num] += coe1 / 2
                    C += coe1 / 4

        for k in range(bs_num):
            for l in range(ue_num):
                for j in range(bs_num):
                    for i in range(ue_num):
                        coe2 = -1 * rb[i, j] * rb[l, k] / bs_num
                        J[i + j * ue_num, l + k * ue_num] += coe2
                        H[i + j * ue_num] += coe2 / 2
                        H[l + k * ue_num] += coe2 / 2
                        C += (coe2 / 4)

        J /= bs_num
        H /= bs_num
        C /= bs_num
        return J, H, C

    @staticmethod
    def data_parse(data_path):
        '''

        :param data_path: .pkl file path
        :return: prb, has shape [time step, UE num, BS num]
        '''
        with open(data_path, "rb") as f:
            coe = pickle.load(f)
            rsrp = coe["RSRP_dBm"]
            sinr = coe["SINR_dB"]
            prb = coe["PRB"]
        return rsrp, sinr, prb

    @staticmethod
    def write_file(file_name, Q):
        f = open(file_name, "w")
        q_size = len(Q)
        ele_num = (q_size * (q_size - 1)) / 2 + q_size
        f.write("%d %d %d\n" % (q_size, q_size, ele_num))
        for i in range(q_size):
            for j in range(i, q_size):
                f.write("%d %d %20.16g\n" % (i, j, Q[i, j]))
        f.close()

    @staticmethod
    def read_file(file_name):
        f = open(file_name, "r")
        data = f.readlines()
        q_size, _, ele_num = data[0].split()
        Q = np.zeros([int(q_size), int(q_size)])
        for n in range(1, int(ele_num) + 1):
            i, j, value = data[n].split()
            Q[int(i), int(j)] = float(value)
            Q[int(j), int(i)] = float(value)
        return Q

    @staticmethod
    def pkl2txt(pkl, save_path):
        with open(pkl, "rb") as f:
            coe = pickle.load(f)
            rsrp = coe["RSRP_dBm"]
            sinr = coe["SINR_dB"]
            prb = coe["PRB"]

            t, ue_num, bs_num = rsrp.shape
            for i in range(0, t, 100):
                data_file = open("{}/t{}.txt".format(save_path, i), "w")
                data_file.write("%d %d\n" % (ue_num, bs_num))
                for ue in range(ue_num):
                    for bs in range(bs_num):
                        data_file.write("%d %d %20.16g %20.16g %20.16g\n"
                                        % (ue, bs, rsrp[i, ue, bs], sinr[i, ue, bs], prb[i, ue, bs]))
                data_file.close()

    @staticmethod
    def read_raw_data(file_name):
        f = open(file_name, "r")
        data = f.readlines()
        ue_num, bs_num = data[0].split()
        rsrp_arr = np.zeros([int(ue_num), int(bs_num)])
        sinr_arr = np.zeros([int(ue_num), int(bs_num)])
        prb_arr = np.zeros([int(ue_num), int(bs_num)])
        for n in range(1, int(ue_num) * int(bs_num) + 1):
            i, j, rsrp, sinr, prb = data[n].split()
            rsrp_arr[int(i), int(j)] = rsrp
            sinr_arr[int(i), int(j)] = sinr
            prb_arr[int(i), int(j)] = prb
        return rsrp_arr, sinr_arr, prb_arr, int(ue_num), int(bs_num)

    @staticmethod
    def check_constrain(binary, constrains):
        for c in constrains:
            r = np.matmul(np.matmul(binary.T, c*binary), binary)
            if r != 0:
                # print(r)
                return False
        return True

    @staticmethod
    def check_Q(Q):
        l = len(Q)
        for i in range(l):
            for j in range(l):
                if Q[i, j] != Q[j, i]:
                    return False
        return True

    @staticmethod
    def init_bin(capacity, q_size, bs_num):
        init_b = np.zeros([q_size])
        for i in range(len(capacity)):
            index = np.where(capacity[i] == np.max(capacity[i]))[0]
            init_b[i*bs_num + index] = 1
        return init_b

    @staticmethod
    def init_serving_list(capacity, ue_num):
        serving_list = np.zeros([ue_num])
        for i in range(len(capacity)):
            index = np.where(capacity[i] == np.max(capacity[i]))[0]
            serving_list[i] = index
        return serving_list


if __name__ == '__main__':
    # pkl_path = "/Users/musktang/Downloads/mlb_data.pkl"
    # # rsrp, sinr, prb = QUBO.data_parse(pkl_path)

    # #### convert pkl file to mmDataFormat txt ####
    # data_path = "/Users/musktang/pycharm_project/mobile-load-balancing/data/raw"
    # QUBO.pkl2txt(pkl_path, data_path)

    #### load raw data from mmDataFormat txt ####
    data_path = "."
    rsrp, capacity, prb, ue_num, bs_num = QUBO.read_raw_data(
        "{}/{}".format(data_path, "small_sample.txt"))

    #### parameters to QUBO matrix ####
    serving_list = QUBO.init_serving_list(capacity, ue_num)
    Q = QUBO.params2qubo_v2(rsrp, capacity, prb,
                            serving_list, ue_num, bs_num, spin=True)

    # print("Q is symmetry : {}".format(QUBO.check_Q(Q[0])))

    # #### write qubo matrix as txt ####
    # file = "/Users/musktang/pycharm_project/mobile-load-balancing/data/jhmatrix/small_sample.txt"
    # QUBO.write_file(file, Q[0])
    # # QUBO.read_file(file)

    init_bin = QUBO.init_bin(capacity, len(Q[0]), bs_num)
    sbm = SBM(Q[0], init_bin, maxStep=1000)
    # sbm = SBM(Q[0], maxStep=1000)
    spin2bin = list()
    for i in sbm.spin:
        spin2bin.append(1 if i > 0 else 0)
    spin2bin = np.expand_dims(np.array(spin2bin), axis=1)
    throughput = np.matmul(np.matmul(spin2bin.T, Q[1]), spin2bin)
    constrain_pass = QUBO.check_constrain(spin2bin, Q[2:])
    print("check constrain pass : {}".format(constrain_pass))
    print("mlb throughput : {}".format(throughput))

    sbm.run()

    spin2bin2 = list()
    for i in sbm.spin:
        spin2bin2.append(1 if i == 1 else 0)
    spin2bin2 = np.expand_dims(np.array(spin2bin2), axis=1)

    throughput = np.matmul(np.matmul(spin2bin2.T, Q[1]), spin2bin2)
    constrain_pass = QUBO.check_constrain(spin2bin2, Q[2:])
    print("check constrain pass : {}".format(constrain_pass))
    print("mlb throughput : {}".format(throughput))
    print("time spent: {}".format(sbm.time))