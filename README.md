# 代码注解
本代码采用tensorflow2.0深度学习框架进行编写。
## 更改数据路径
下载完本代码之后，先看到code文件夹里的config文件，其中的train_label_path,test_path和train_label_path2,test_path2需要改写为自己的数据路径。同时在make_noise.py和make_noise2.py文件的train_path,test_path里也需要修改。
## make_noise.py文件与make_noise2.py文件
train_noise_path和train_noise_path2两个路径下存放有添加噪声后的part1和part2的OK_Images图像（各1000张），这些图像通过make_noise.py,make_noise2.py这两个文件即可生成。
## train.py文件
在运行train.py来训练模型之前需要在model.py文件下确认AES与DS模型中bn层trainable参数为True（在进行测试时trainable为False），通过运行train.py文件即可得到模型的h5文件，得到的模型文件可以在model文件夹里找到。
## demo.py文件
在code文件夹通过运行demo.py文件可以查看模型对测试图像的重建效果和与原图对比得到的缺陷图像，按空格键可以查看下一张，按ESC键退出。
## main.py
通过运行code文件夹下的main.py文件可以在同目录下的data文件夹中生成json文件。
## 一些问题
当我用这个网络训练part2的数据时，重建效果不是很理想，对比part1的重建效果差很多，之后可以在模型上进行改进。