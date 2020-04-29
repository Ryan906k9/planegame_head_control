# planegame_head_control
Use your head to control a plane in the game

在飞机大战游戏的基础上，用人脸识别作为控制器！

## 第2版修正版已经上传，文件为 main_2.py

1. 大幅提升流畅度
2. 应用新的头部角度计算算法，缩减代码量100+
3. 修改背景，音乐等



## 文件夹：
bgimages：存放背景图片

boss：boss相关图片

font：字体文件

images：飞机，子弹等图片文件

sound：声音文件

## 文件：
_plane.py 补给飞机

bullet.py 子弹

enemy_bullet.py 敌机子弹

enemy.py 敌机

main_origin.py 原飞机大战主文件

main.py 添加了人脸识别作为控制器的飞机大战主文件

plane.py 飞机

pose_estimation_enhancement.py 人脸识别程序，这里作为参考用，游戏中未调用

shield.py 护盾

supply.py 奖励

## 需要安装相应的包：
pip install pygame

pip install paddlehub

## 运行方式：
直接运行 main.py 即可

头部左转：飞机往左移动

头部右转：飞机往右移动

抬头：飞机前移动

低头：飞机向后移动

张嘴：丢炸弹！


## 说明：
飞机的速度，子弹的速度都可以在参数中调节

为了演示方便，把之前游戏中第一关的速度都上调了

由于摄像头的镜像关系，头部左转和右转会与左右控制相反，希望调节为反过来的可以在参数里对调一下

