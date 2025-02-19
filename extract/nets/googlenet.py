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
from PIL import Image
from torchvision.models import googlenet, GoogLeNet_Weights


def extract_googlenet(figpath):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 检查文件是否存在
        if not os.path.isfile(figpath):
            raise FileNotFoundError("File does not exist")
        model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        # 修改模型以用于特征提取
        # 移除辅助输出和最后的全连接层
        model.aux_logits = False
        model.fc = torch.nn.Identity()
        # 将模型设置为评估模式
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 加载图像
        img = read_image(figpath).convert('RGB')  # 替换为你的图像路径
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


def extract_googlenet_content(image):
    """ 提取图像的GoogLeNet特征 """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 检查输入是否为PIL图像并转换为NumPy数组
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是PIL图像对象")

        # 加载预训练的 GoogLeNet 模型
        model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        # 修改模型以用于特征提取
        # 移除辅助输出和最后的全连接层
        model.aux_logits = False
        model.fc = torch.nn.Identity()
        # 将模型设置为评估模式
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 预处理图像
        img = image.convert('RGB')
        img_t = preprocess(img)
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
