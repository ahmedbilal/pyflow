# Author: Deepak Pathak (c) 2016

import numpy as np
import cv2
import pyflow.pyflow as pyflow


def optical_flow(img1, img2):
    im1 = img1.astype(float) / 255.
    im2 = img2.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    min_width = 20
    n_outer_fp_iterations = 7
    n_inner_fp_iterations = 1
    n_sor_iterations = 30
    col_type = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, min_width, n_outer_fp_iterations, n_inner_fp_iterations,
        n_sor_iterations, col_type)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb
