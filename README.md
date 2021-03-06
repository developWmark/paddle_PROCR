# paddle_PROCR 论文复现
## 论文：Primitive Representation Learning for Scene Text Recognition
## 论文地址：https://openaccess.thecvf.com/content/CVPR2021/html/Yan_Primitive_Representation_Learning_for_Scene_Text_Recognition_CVPR_2021_paper.html

## 1.简介：
   在此非常感谢论文原作者的参考代码 https://github.com/RuijieJ/pren ，提高了本repo复现论文的效率。
   论文作者提出了一种原始表示学习方法，旨在利用场景文本图像的内在表示。作者将特征映射中的元素建模为无向图的节点，提出了一种池聚合器和加权聚合器来学习原始表示，
   并通过图卷积网络将原始表示转化为高级视觉文本表示。构造了一个原始表示学习网络（PREN）来使用视觉文本表示进行并行解码。此外，通过将视觉文本表示集成到具有2D注意机制的编码器模型中，
   提出了一个名为PREN2D的框架，以缓解基于注意的方法中的错位问题。在英语和汉语场景文本识别任务上的实验结果表明，PREN在准确性和效率之间保持平衡，而PREN2D达到了最先进的性能。
   本代码复现的PREN框架。
   
   ![image](https://github.com/developWmark/paddle_PROCR/blob/master/samples/framework.png)
 
##  2.复现精度：
   ![image](https://github.com/developWmark/paddle_PROCR/blob/master/samples/result.png)
 
##  3.权重和测试数据集：
    链接: https://pan.baidu.com/s/1WlyW-csVjV3VjvjLUygOTg 提取码: q0bw 
 
    将百度网盘的文件夹放在项目根目录下，为了提高训练效果，采用的lmdb数据存储格式。
   ![image](https://github.com/developWmark/paddle_PROCR/blob/master/samples/show1.png)
   
## 4.环境依赖
    pip install -r ./requirement.txt
## 5快速开始   
###  5.1 训练：
       1.将数据集放入dataset文件夹下（数据集来自https://github.com/FangShancheng/ABINet，在readme.md下载mj+st数据集，
       已上传至aistudio：https://aistudio.baidu.com/aistudio/datasetdetail/120703）
   
       2.在项目根目录下创建dataset，并且将数据集解压到dataset下，并且在config.py文件下修改对应的位置 如：_C.LMDB.trainData_dir = "./dataset/lmdb/training"
   ![image](https://github.com/developWmark/paddle_PROCR/blob/master/samples/Screenshot_select-area_20220207160211.png)
   
       3.pip install -r ./requirement.txt (依赖基于aistudio环境)
       
       4.python ./main_single_gpu.py （单卡大概需要6天）
       
       
###  5.2 测试（lmdb） ：

        1.网盘中的dataset内有测试数据集 ,测试数据集来自：https://aistudio.baidu.com/aistudio/datasetdetail/114635
        2.在config.py 文件中修改对应的位置 如：_C.LMDB.testDir="./dataset/evaluation/SVT"
        3.在config.py 文件中修改权重加载路径 _C.LMDB.testResume='./output/resume2/Epoch-8-Loss-0.18709387933158853'     
        4.python ./test.py

###  5.3 测试（single img）：
         1. python ./testSingle.py --filepath='samples/001.jpg'
         预测图片：
   ![image](https://github.com/developWmark/paddle_PROCR/blob/master/samples/001.jpg)
         
         预测结果：
   ![image](https://github.com/developWmark/paddle_PROCR/blob/master/samples/Screenshot_select-area_20220209210840.png)
         


         
       
       
    
