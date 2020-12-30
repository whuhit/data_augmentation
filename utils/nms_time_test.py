"""
https://github.com/SirLPS/NMS
https://www.cnblogs.com/king-lps/p/9031568.html
"""
import numpy as np
import time
from nms.nms_cy import py_cpu_nms  # for cpu
from nms.nms_py import py_nms


np.random.seed(1)  # keep fixed
num_rois = 6000
minxy = np.random.randint(50, 145, size=(num_rois, 2))
maxxy = np.random.randint(150, 200, size=(num_rois, 2))
score = 0.8 * np.random.random_sample((num_rois, 1)) + 0.2
boxes_new = np.concatenate((minxy, maxxy, score), axis=1).astype(np.float32)

def nms_test_time(boxes_new,f=None):
    thresh = [0.7, 0.8, 0.9]
    T = 50
    for i in range(len(thresh)):
        since = time.time()
        for t in range(T):
            keep = f(boxes_new, thresh=thresh[i])  # for cpu
        print("thresh={:.1f}, time wastes:{:.4f}".format(thresh[i], (time.time() - since) / T))
    return keep

if __name__ == "__main__":
    nms_test_time(boxes_new, f=py_cpu_nms)
    print()
    nms_test_time(boxes_new, f=py_nms)
    # py_cpu_nms
    # thresh=0.7, time wastes:0.0019
    # thresh=0.8, time wastes:0.0026
    # thresh=0.9, time wastes:0.0034
    # py_nms
    # thresh=0.7, time wastes:0.0325
    # thresh=0.8, time wastes:0.1221
    # thresh=0.9, time wastes:0.4838