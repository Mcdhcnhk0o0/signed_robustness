import numpy as np
import matplotlib.pyplot as plt
import balance.balance_utils as but
from common.file_operations import FileOperations


def f_seq2r_seq(f_seq: np.ndarray, M: int):
    bgs = (M - f_seq) / M
    rbs = np.zeros(shape=f_seq.shape)

    cur_rb = 0
    n = len(rbs)

    for i in range(n):
        cur_rb += bgs[i]
        rbs[i] = cur_rb / (i + 1)

    return rbs


def get_real_frustrations(f_seq, p=1.3, s=0, e=500):

    n = len(f_seq)

    for i in range(2, n-2):
        if i < s or i > e:
            continue
        window = [f_seq[i-1], f_seq[i], f_seq[i+1]]
        if f_seq[i] > min(window) * p:
            f_seq[i] = min(window)

    return f_seq


dataset_path = r"F:\github\structural_balance_ig-main\datasets/synthetic_for_robustness/syn_network_5_100_12_0.6_0.0_0"
ds_info = but.get_dataset_info(dataset_path)
M98 = min(ds_info['positive enum'], ds_info['negative enum'])

f_res = FileOperations.load_array_from_file("process_0.6_betweenness.txt")
# f_res = np.array([54])
n = len(f_res)

plt.plot(range(n), f_res)
plt.show()

rf = get_real_frustrations(f_res, 1.5, 0, 440)
rf = get_real_frustrations(rf, 1.4)
rf = get_real_frustrations(rf, 1.15, 50, 410)
# rf = get_real_frustrations(rf, 1.4)

plt.plot(range(n), rf)
plt.show()

r_res = f_seq2r_seq(rf, M98)
print(r_res)

for i in range(10):
    print(i, r_res[int(0.1 * i * n)])
