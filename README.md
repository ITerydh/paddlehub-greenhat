# paddlehub-greenhat
一个图片带绿帽子的项目

# 利用PaddleHub戴绿帽[人间的青草地需要浇水~]

被抖音洗脑的人间的青草地需要浇水~

驱使了我想要做本项目
附AIstudio项目链接 

> [aistudio 项目地址https://aistudio.baidu.com/aistudio/projectdetail/2108944](https://aistudio.baidu.com/aistudio/projectdetail/2108944)

话不多说，开干！




```python
# 安装Paddlehub
!pip install paddlehub==2.1.0
```


```python
! hub install ultra_light_fast_generic_face_detector_1mb_640==1.1.2
```

## 通过paddlehub获取位置信息


```python
import paddlehub as hub

module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640", version="1.1.2")

res = module.face_detection(
    paths = ["data/data96296/1.jpg"],
    visualization=True,
    output_dir='face_detection_output')

```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized
    [2021-06-22 16:33:08,556] [ WARNING] - The _initialize method in HubModule will soon be deprecated, you can use the __init__() to handle the initialization of the object


## 环境准备


```python
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

filename = 'face_detection_output/1.jpg'
## [Load an image from a file]
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2366: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return list(data) if isinstance(data, collections.MappingView) else data





    <matplotlib.image.AxesImage at 0x7fe05c0632d0>




![png](https://github.com/ITerydh/paddlehub-greenhat/blob/main/output_6_2.png)



```python
res
# 返回
# res (list[dict]): 识别结果的列表，列表中每一个元素为 dict，关键字有'path', 'data'，相应的取值为：
# path (str): 原输入图片的路径；
# data (numpy.ndarray): 图像分割得到的结果，shape 为H * W，元素的取值为0-19，表示每个像素的分类结果，映射顺序与下面的调色板相同。
```




    [{'data': [{'left': 221.29409790039062,
        'right': 488.7735290527344,
        'top': 78.7082290649414,
        'bottom': 471.13763427734375,
        'confidence': 0.9999988079071045}],
      'path': 'data/data96296/1.jpg',
      'save_path': 'face_detection_output/1.jpg'}]




```python
# 处理为整数，小数不符合后续位置参数的要求
import math

data = res[0]['data']
left = data[0]['left']
left = math.ceil(left)
right = data[0]['right']
right = math.ceil(right)
top = data[0]['top']
top = math.ceil(top)
bottom = data[0]['bottom']
bottom = math.ceil(bottom)
print("left:",left)
print("right:",right)
print("top:",top)
print("bottom:",bottom)
```

    left: 222
    right: 489
    top: 79
    bottom: 472


有上面的数据可以知道:

框的左上角顶点位置为[left,top]

框的右上角顶点位置为[right,top]

得到这些信息后，再带上提前做好的帽子，就思路清晰了

将帽子尺寸改为[right-left+25,2.2*top]   25和2.2可以自己进行调整，合适为止。


```python
# 储存帽子数据
hat_w , hat_h=  right-left+40, 2.2*top
hat_w = math.ceil(hat_w)
hat_h = math.ceil(hat_h)
print(hat_w)
print(hat_h)
```

    307
    174


## 图片合成


```python
from PIL import Image

mother_img = "data/data96296/1.jpg"
son_img = "data/data96296/lmz.png"
save_img = "result/result.png"

# 获取图片,方便后面的代码调用
M_Img = Image.open(mother_img)
S_Img = Image.open(son_img)

# 子图缩小的倍数1代表不变，2就代表原来的一半（没啥用，本来想通过这个控制图片）
factor = 1

# 给图片指定色彩显示格式
S_Img = S_Img.convert("RGBA")

# 获取图片的尺寸
M_Img_w, M_Img_h = M_Img.size  # 获取被放图片的大小（母图）
print("母图尺寸：",M_Img.size)
S_Img_w, S_Img_h = S_Img.size  # 获取小图的大小（子图）
print("子图尺寸：",S_Img.size)

size_w = int(S_Img_w / factor)
size_h = int(S_Img_h / factor)

# 防止子图尺寸大于母图
if S_Img_w > size_w:
    S_Img_w = size_w
if S_Img_h > size_h:
    S_Img_h = size_h

# 重新设置子图的尺寸
icon = S_Img.resize((hat_w, hat_h), Image.ANTIALIAS)

# 子图的左上角坐标位置
w = left-20
h = 0

# 粘贴子图到母图的指定坐标（当前居中）
M_Img.paste(icon, (w, h), mask=icon)

# 保存图片
M_Img.save(save_img)
```

    母图尺寸： (693, 539)
    子图尺寸： (224, 236)


## 查看效果


```python
filename = 'result/result.png'
## [Load an image from a file]
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x7fe050429050>




![png](https://github.com/ITerydh/paddlehub-greenhat/blob/main/6.png)


# 总结

1. 利用PaddleHub获取人体的头部位置
2. 合理计算出帽子所需要放置的位置
3. 图片合成
x(已解决) 4. 很明显（我这个版本很失败，还没有找到好的办法去把图片的黑色去掉，郁闷（明明帽子图片是PNG且代码也写了RGBA））

我在AI Studio上获得钻石等级，点亮9个徽章，来互关呀~ 

https://aistudio.baidu.com/aistudio/personalcenter/thirdview/643467

