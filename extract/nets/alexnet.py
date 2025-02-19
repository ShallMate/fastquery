# Copyright 2025 Guowei LING.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from read.readfig import read_image
import logging
import os
from torchvision.transforms import Lambda
from PIL import Image
from torchvision.models import alexnet, AlexNet_Weights


def extract_alexnet(figpath):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 检查文件是否存在
        if not os.path.isfile(figpath):
            raise FileNotFoundError("File does not exist")
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        # 移除最后的全连接层，以提取特征而不进行分类
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

        # 图像预处理步骤
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            Lambda(lambda x: x.convert('RGB')),  # 转换为RGB
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225]),
        ])
        # 加载图像
        img = read_image(figpath)  # 替换为你的图像路径
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0)

        # 提取特征
        with torch.no_grad():
            features = model(batch_t)

        return features

    except FileNotFoundError as e:
        logging.error("Error: %s", e)
        return None
    except Exception as e:
        logging.error("An error occurred: %s", e)
        return None


def extract_alexnet_content(image):
    """ 提取图像的AlexNet特征 """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 检查输入是否为PIL图像并转换为NumPy数组
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是PIL图像对象")

        # 加载预训练的 AlexNet 模型
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        # 移除最后的全连接层，以提取特征而不进行分类
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

        # 图像预处理步骤
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            Lambda(lambda x: x.convert('RGB')),  # 转换为RGB
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225]),
        ])

        # 预处理图像
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)

        # 提取特征
        with torch.no_grad():
            features = model(batch_t)

        return features

    except ValueError as e:
        logging.error("Error: %s", e)
        return None
    except Exception as e:
        logging.error("An error occurred: %s", e)
        return None
