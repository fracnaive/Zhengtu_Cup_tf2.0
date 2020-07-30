import numpy as np
import cv2
import os
import random

train_path = 'D:/zhengtu_cup/data/focusight1_round1_train_part1/OK_Images'
test_path = 'D:/zhengtu_cup/data/focusight1_round1_train_part1/TC_Images'
random.seed(25)
np.random.seed(25)


# 第一种缺陷生成函数，在灰色区域(样本区域)生成随机大小，亮度(像素值)为高亮(接近255)，或者为灰暗(接近0)
# gray_src的shape为(128, 128)，灰度图像
# value为像素点值的选取范围，类型为list
# 随机选取的比例，float
def defect1(gray_src1, binary, value1, ratio1):

    gray_src = np.uint8(np.copy(gray_src1))
    binary_src = np.uint8(np.copy(binary))
    # 寻找轮廓，一般只会出现一个轮廓，若不是则报错
    contour1, _ = cv2.findContours(binary_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contour1) == 1

    # 设置噪声区域大小选取范围
    sizes1 = range(10, 20)
    # 从中随机选取噪声大小
    choice_size1 = random.choice(sizes1)

    # 设置噪声区域中心点位置范围
    inner_points1 = []
    for i1 in range(128):
        for j1 in range(128):
            if binary_src[i1, j1] == 255:
                inner_points1.append((i1, j1))
    # 从该范围中随机选取噪声中心点
    choice_point1 = random.choice(inner_points1)

    # 计算边界点，并进行限幅
    b1 = int(choice_size1 / 2)
    x_min1 = choice_point1[0] - b1
    x_max1 = choice_point1[0] + b1
    y_min1 = choice_point1[1] - b1
    y_max1 = choice_point1[1] + b1
    if x_min1 < 0:
        x_min1 = 0
    if x_max1 >= 128:
        x_max1 = 127
    if y_min1 < 0:
        y_min1 = 0
    if y_max1 >= 128:
        y_max1 = 127

    # 如果选取的噪声区域的点在样本上，则选为缺陷点
    defect_points1 = []
    for x1 in range(x_min1, x_max1 + 1):
        for y1 in range(y_min1, y_max1 + 1):
            flag1 = cv2.pointPolygonTest(contour1[0], (y1, x1), False)
            if flag1 >= 0:
                defect_points1.append((x1, y1))

    # 从选取的缺陷点中随机剔除20%的缺陷点
    num1 = len(defect_points1)
    defect_points1 = random.sample(defect_points1, int(ratio1*num1))

    # 修改噪声点的像素值
    for point1 in defect_points1:
        gray_src.itemset(point1, random.choice(value1))

    return gray_src


def defect2(gray_src2, binary, value, ratio):

    gray_src = np.uint8(np.copy(gray_src2))
    binary_src = np.uint8(np.copy(binary))
    # 寻找轮廓，一般只会出现一个轮廓，若不是则报错
    contour, _ = cv2.findContours(binary_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contour) == 1

    M = cv2.moments(contour[0])
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    choice_point = (center_y, center_x)

    cv2.circle(gray_src, (center_x, center_y), 3, 128, -1)

    # 设置噪声区域中心点位置范围
    inner_points = []
    for i in range(128):
        for j in range(128):
            if binary_src[i, j] == 255:
                inner_points.append((i, j))

    x = range(0, 127)

    a = range(0, 100)
    choice_a = random.choice(a) / 10.0
    choice_b = choice_point[1] - choice_a * choice_point[0]

    choice_d = 20
    choice_r = int(choice_d / 2)

    defect_points = []
    for i in x:
        j = int(choice_a * i + choice_b)
        if j >= 128:
            continue
        j = range(j - 3, j + 3)
        j = random.choice(j)
        if j < 0 or j >= 128:
            continue
        for rr in range(-choice_r, choice_r+1):
            y = j + rr
            if y < 0 or y >= 128:
                continue
            flag = cv2.pointPolygonTest(contour[0], (y, i), False)
            if flag >= 0:
                defect_points.append((i, y))

    # x_min = np.min(defect_points[:, 0])
    defect_points = rand_len_defect2(defect_points)
    num = len(defect_points)
    defect_points = random.sample(defect_points, int(ratio * num))

    for point in defect_points:
        gray_src.itemset(point, random.choice(value))

    return gray_src


def rand_len_defect2(defect_p):
    x_min = 128
    x_max = 0
    defect_p1 = []
    for (x, y) in defect_p:
        if x < x_min:
            x_min = x
        if x > x_max:
            x_max = x
    x_range = range(x_min, x_max)
    if len(x_range) >= 2:
        x_ba = random.sample(x_range, 2)
        if x_ba[0] > x_ba[1]:
            x_pa_min = x_ba[1]
            x_pa_max = x_ba[0]
        else:
            x_pa_min = x_ba[0]
            x_pa_max = x_ba[1]

        for (x, y) in defect_p:
            if x < x_pa_min or x > x_pa_max:
                continue
            defect_p1.append((x, y))
    else:
        defect_p1 = defect_p
    return defect_p1


def add_gaussian_noise(gray_src, binary, sigma):
    temp = np.uint8(np.copy(gray_src))
    binary_src = np.uint8(np.copy(binary))
    # 寻找轮廓，一般只会出现一个轮廓，若不是则报错
    contour, _ = cv2.findContours(binary_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contour) == 1

    h, w = temp.shape

    noise = np.random.randn(h, w) * sigma
    noise_image = temp + noise

    for i in range(h):
        for j in range(w):
            flag = cv2.pointPolygonTest(contour[0], (j, i), False)
            if flag >= 0:
                temp.itemset((i, j), noise_image[i][j])

    return temp


square = [0, 1, 2]
line = [0, 1, 2]
gauss = [0, 1]
wb = [0, 1]
for i in range(1, 1001):
    square_r = random.choice(square)
    line_r = random.choice(line)
    gauss_r = random.choice(gauss)
    # wb_r = random.choice(wb)

    src = cv2.imread(os.path.join(train_path, 'Image_'+str(i)+'.bmp'))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 二值化小于50的为0，大于50的为255
    _, b_src = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    val = range(8, 50)
    val1 = range(128, 256)

    if square_r == 1:
        wb_r = random.choice(wb)
        if wb_r == 0:
            wb_val = val
        else:
            wb_val = val1
        defect_gray1 = defect1(gray, b_src, value1=wb_val, ratio1=0.8)
    elif square_r == 2:
        wb_r = random.choice(wb)
        if wb_r == 0:
            wb_val = val
        else:
            wb_val = val1
        defect_gray1 = defect1(gray, b_src, value1=wb_val, ratio1=0.8)
        wb_r = random.choice(wb)
        if wb_r == 0:
            wb_val = val
        else:
            wb_val = val1
        defect_gray1 = defect1(defect_gray1, b_src, value1=wb_val, ratio1=0.8)
    else:
        defect_gray1 = gray

    if line_r == 1:
        wb_r = random.choice(wb)
        if wb_r == 0:
            wb_val = val
        else:
            wb_val = val1
        defect_gray2 = defect2(defect_gray1, b_src, value=wb_val, ratio=1)
    elif line_r == 2:
        wb_r = random.choice(wb)
        if wb_r == 0:
            wb_val = val
        else:
            wb_val = val1
        defect_gray2 = defect2(defect_gray1, b_src, value=wb_val, ratio=1)
        wb_r = random.choice(wb)
        if wb_r == 0:
            wb_val = val
        else:
            wb_val = val1
        defect_gray2 = defect2(defect_gray2, b_src, value=wb_val, ratio=1)
    else:
        defect_gray2 = defect_gray1

    if gauss_r == 1:
        gaussian = add_gaussian_noise(defect_gray2, b_src, 25)
    else:
        gaussian = defect_gray2

    cv2.imwrite('../temp_data/part1_noise_images/image_{}.bmp'.format(str(i)), gaussian)
    print('making:\t{}/1000'.format(str(i)))

# while 1:
#     # cv2.imshow('d1', defect_gray1)
#     # cv2.imshow('d2', defect_gray2)
#     cv2.imshow('g', gaussian)
#     if cv2.waitKey(10) == 27:
#         break
