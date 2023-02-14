# 作业3
## 作业要求
使用MMSegmentation，在自己的数据集上，训练语义分割模型
数据集标注（可选）
使用Labelme、LabelU等数据标注工具，标注多类别语义分割数据集，并保存为指定的格式。

数据集整理
划分训练集、测试集

使用MMSegmentation训练语义分割模型
在MMSegmentation中，指定预训练模型，配置config文件，修改类别数、学习率。

用训练得到的模型预测
获得测试集图片或新图片的语义分割预测结果，对结果进行可视化和后处理。

在测试集上评估算法的速度和精度性能

使用MMDeploy部署语义分割模型（可选）

## 作业汇报

模型来自 pspnet_r50-d8_4xb2-40k_cityscapes-512x1024
数据集来自
![斯坦福官网数据集子豪兄加持版](https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/iccv09Data.tar.gz -O stanford_background.tar.gz)
```sh
bash run_seg.sh
```
- run_seg.sh
安装mmsegmentation环境并训练
- train.py
训练内容
- MMSeg2.png
6000126.png 训练1600iter后的预测(待优化)

模型链接：
