# openvino-YOLOv5
利用openvino框架的YOLOv5的轻量化部署，在windows端（linux端同理）

github地址：https://github.com/wang-yichen/openvino-YOLOv5/tree/main

## 1.openvino开发工具下载

可查阅官方文档（https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-windows.html），查看openvino环境的安装部分，本文使用的是c++版本开发YOLOv5部署，使用的是存档安装的方法，具体前往官网：

https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_4_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=PIP

下载最新的安装包，如下图所示：

![1727140971850](D:/best/openvino/1727140971850.png)

点击Download下载zip包。

下载好后解压，放到任意位置，向环境变量path中添加如下变量，按照自己的位置修改

```
D:\Program Files\openvino\openvino_2024\runtime\bin\intel64\Debug
D:\Program Files\openvino\openvino_2024\runtime\bin\intel64\Release
D:\Program Files\openvino\openvino_2024\runtime\3rdparty\tbb\bin
```



## 2.python环境安装

这里选择使用conda的虚拟环境安装openvino的python环境

```
pip install openvino-genai==2024.3.0
```

根据YOLOv5需要，安装下列安装包

```cmd
python -m pip install --upgrade pip
pip install onnx
pip install torch
pip install mxnet
```

先从YOLOv5的环境中将.pt转换为.onnx模型，版本选择10

随后编写如下脚本

```python
from openvino.runtime import Core
from openvino.runtime import serialize
 
ie = Core()
onnx_model_path = r"自己的模型路径.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
# compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
serialize(model=model_onnx, xml_path="生成xml的模型路径.xml", bin_path="生成bin模型的路径.bin", version="UNSPECIFIED")
```

运行后生成.xml和.bin模型，如果报错说明环境安装有问题

## 3.c++部署

创建新项目，release版本

包含目录添加如下内容

```
D:\Program Files\openvino\openvino_2024\runtime\bin\intel64\Release
D:\Program Files\openvino\openvino_2024\runtime\include
D:\Program Files\openvino\openvino_2024\runtime\include\openvino
C:\E\Users\wangyichen\opencv\build\include\opencv2
C:\E\Users\wangyichen\opencv\build\include
```

库目录添加如下内容

```
C:\E\Users\wangyichen\opencv\build\x64\vc16\lib
D:\Program Files\openvino\openvino_2024\runtime\lib\intel64\Release
```

链接器附加依赖项添加如下内容（依照自己的opencv版本添加）

```
opencv_world470.lib
openvino.lib
```

如下为代码部分,直接添加进自己的项目中

```
openvino.cpp
openvino.h
main.cpp
```

如此运行可以得到正确的结果，如果有报错请在评论区提出来