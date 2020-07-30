import numpy as np
import cv2
import os
import random
from numba import jit

train_path = 'D:/zhengtu_cup/data/focusight1_round1_train_part2/OK_Images'
test_path = 'D:/zhengtu_cup/data/focusight1_round1_train_part2/TC_Images'
random.seed(25)
np.random.seed(25)


def main():
    for i in range(1, 1001):
        path = os.path.join(train_path, 'Image_'+str(i)+'.bmp')
        src = cv2.imread(path)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # print(gray.shape)
        # 二值化小于50的为0，大于50的为255
        _, b_src = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        @jit  # python加速，返回边缘点坐标l(左坐标点集合)，r(右坐标点集合)
        def edge(image):
            l = np.full(128*128, -1)
            r = np.full(128*128, -1)
            n = 0
            for ii in range(128):
                for jj in range(128):
                    if ii == 0 or jj == 0 or ii == 127 or jj == 127:
                        continue
                    v = (int(image[ii][jj-1]) + int(image[ii][jj+1]) + int(image[ii-1][jj]) + int(image[ii+1][jj])) / 4
                    if image[ii][jj] <= 50 and v >= 50:
                        l[n] = ii
                        r[n] = jj
                        n += 1
            return l, r
        l, r = edge(gray)
        # 将得到的左右坐标点集合转换为坐标集合
        e = []
        for p in range(len(l)):
            if l[p] == -1:
                break
            e.append((l[p], r[p]))
        # print(e)

        def edge_circle(gry, color):
            edge_list_num = list(range(len(e)))
            edge_point = list(e[random.choice(edge_list_num)])
            black = list(range(50))
            white = list(range(200, 255))
            edge_circle_r = random.choice(list(range(10, 20)))
            temp = np.uint8(np.copy(gry))
            for ai in range(128):
                for aj in range(128):
                    r = (ai - edge_point[0])**2 + (aj - edge_point[1])**2
                    if r < edge_circle_r**2:
                        if color == 'black':
                            if b_src[ai][aj] <= 50:
                                temp.itemset((ai, aj), gry[ai][aj])
                            else:
                                temp.itemset((ai, aj), random.choice(black))
                        elif color == 'white':
                            if b_src[ai][aj] <= 50:
                                temp.itemset((ai, aj), random.choice(white))
                            else:
                                temp.itemset((ai, aj), gry[ai][aj])
                    else:
                        temp.itemset((ai, aj), gry[ai][aj])
            return temp

        def edge_square(gry, color):
            edge_list_num = list(range(len(e)))
            edge_point = list(e[random.choice(edge_list_num)])
            black = list(range(50))
            white = list(range(250, 255))
            temp = np.uint8(np.copy(gry))
            square_size = random.choice(list(range(8, 16)))
            for ai in range(128):
                for aj in range(128):
                    if abs(ai-edge_point[0]) < square_size and abs(aj-edge_point[1]) < square_size:
                        if color == 'black':
                            if b_src[ai][aj] <= 50:
                                temp.itemset((ai, aj), gry[ai][aj])
                            else:
                                temp.itemset((ai, aj), random.choice(black))
                        elif color == 'white':
                            temp.itemset((ai, aj), random.choice(white))
                    else:
                        temp.itemset((ai, aj), gry[ai][aj])
            return temp

        def inner_circle(gry):
            edge_list_num = list(range(len(e)))
            edge_point = list(e[random.choice(edge_list_num)])
            white = list(range(100, 255))
            inner_x_points1 = list(range(edge_point[0], 128))
            inner_x_points2 = list(range(edge_point[0]))
            inner_x_point1 = random.choice(inner_x_points1)
            inner_x_point2 = random.choice(inner_x_points2)
            if gry[inner_x_point1][edge_point[1]] <= 50:
                inner_x_point = inner_x_point1
            else:
                inner_x_point = inner_x_point2
            inner_circle_r = random.choice(list(range(5, 12)))
            temp = np.uint8(np.copy(gry))
            for ai in range(128):
                for aj in range(128):
                    r = (ai - inner_x_point)**2 + (aj - edge_point[1])**2
                    if r < inner_circle_r**2:
                        if b_src[ai][aj] <= 50:
                            temp.itemset((ai, aj), random.choice(white))
                        else:
                            temp.itemset((ai, aj), gry[ai][aj])
                    else:
                        temp.itemset((ai, aj), gry[ai][aj])
            return temp

        def inner_square(gry):
            edge_list_num = list(range(len(e)))
            edge_point = list(e[random.choice(edge_list_num)])
            white = list(range(100, 255))
            inner_x_points1 = list(range(edge_point[0], 128))
            inner_x_points2 = list(range(edge_point[0]))
            inner_x_point1 = random.choice(inner_x_points1)
            inner_x_point2 = random.choice(inner_x_points2)
            if gry[inner_x_point1][edge_point[1]] <= 50:
                inner_x_point = inner_x_point1
            else:
                inner_x_point = inner_x_point2
            temp = np.uint8(np.copy(gry))
            square_size = random.choice(list(range(8, 12)))
            for ai in range(128):
                for aj in range(128):
                    if abs(ai-inner_x_point) < square_size and abs(aj-edge_point[1]) < square_size:
                        if b_src[ai][aj] <= 50:
                            temp.itemset((ai, aj), random.choice(white))
                        else:
                            temp.itemset((ai, aj), gry[ai][aj])
                    else:
                        temp.itemset((ai, aj), gry[ai][aj])
            return temp

        def inner_lines(gry):
            edge_list_num = list(range(len(e)))
            edge_point = list(e[random.choice(edge_list_num)])
            white = list(range(100, 255))
            inner_x_points1 = list(range(edge_point[0], 128))
            inner_x_points2 = list(range(edge_point[0]))
            inner_x_point1 = random.choice(inner_x_points1)
            inner_x_point2 = random.choice(inner_x_points2)
            if gry[inner_x_point1][edge_point[1]] <= 50:
                inner_x_point = inner_x_point1
            else:
                inner_x_point = inner_x_point2
            temp = np.uint8(np.copy(gry))
            coefficient_b = random.choice(range(-256, 256))
            coefficient_a = float((inner_x_point - coefficient_b)) / edge_point[1]
            for ai in range(128):
                for aj in range(128):
                    d = abs(coefficient_a*aj - ai + coefficient_b)/(np.sqrt(coefficient_a**2 + 1))
                    if d < 2:
                        if b_src[ai][aj] <= 50:
                            temp.itemset((ai, aj), random.choice(white))
                        else:
                            temp.itemset((ai, aj), gry[ai][aj])
                    else:
                        temp.itemset((ai, aj), gry[ai][aj])
            return temp

        color_num1 = random.choice(list(range(2)))
        if color_num1 == 0:
            tmp = edge_circle(gray, color='black')
        else:
            tmp = edge_circle(gray, color='white')

        tmp1 = inner_circle(tmp)
        add_num = random.choice(list(range(3)))
        if add_num == 0:
            tmp2 = tmp1
        elif add_num == 1:
            tmp2 = inner_lines(tmp1)
        else:
            tmp2 = inner_lines(tmp1)
            tmp2 = inner_lines(tmp2)
        color_num2 = random.choice(list(range(2)))
        if color_num2 == 0:
            tmp3 = edge_square(tmp2, color='black')
        else:
            tmp3 = edge_square(tmp2, color='white')
        tmp4 = inner_square(tmp3)

        cv2.imwrite('../temp_data/part2_noise_images/image_{}.bmp'.format(str(i)), tmp4)
        print('making:\t{}/1000'.format(str(i)))
        # cv2.imshow('aaa', b_src)
        # cv2.imshow('bbb', tmp)
        # cv2.imshow('ccc', tmp1)
        # cv2.imshow('ddd', tmp2)
        # cv2.imshow('eee', tmp3)
        # cv2.imshow('fff', tmp4)
        # while 1:
        #     if cv2.waitKey() == 27:
        #         break


if __name__ == '__main__':
    main()
