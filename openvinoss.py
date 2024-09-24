from openvino.runtime import Core
from openvino.runtime import serialize

import os

# 检查目录是否存在，如果不存在则创建
dir_path = r"D:/best"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

ie = Core()
onnx_model_path = r"D:/best/best.onnx"
model_onnx = ie.read_model(model=onnx_model_path)
# compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
serialize(model=model_onnx, xml_path="D:/best/best.xml", bin_path="D:/best/best.bin", version="UNSPECIFIED")