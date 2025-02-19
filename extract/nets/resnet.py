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
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18, resnet50, resnet152, ResNet18_Weights, ResNet50_Weights, ResNet152_Weights


def extract_resnet18(figpath):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 检查文件是否存在
        if not os.path.isfile(figpath):
            raise FileNotFoundError("File does not exist")

        # 加载预训练的 ResNet18 模型
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # 移除最后的全连接层，以提取特征而不进行分类
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

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


def extract_resnet50(figpath):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 检查文件是否存在
        if not os.path.isfile(figpath):
            raise FileNotFoundError("File does not exist")

        # 加载预训练的 ResNet50 模型
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # 移除最后的全连接层，以提取特征而不进行分类
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()
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


def extract_resnet152(figpath):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 检查文件是否存在
        if not os.path.isfile(figpath):
            raise FileNotFoundError("File does not exist")

        # 加载预训练的 ResNet152 模型
        model = resnet152(weights=ResNet152_Weights.DEFAULT)
        # 移除最后的全连接层，以提取特征而不进行分类
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

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


def extract_resnet50_specific_layers(figpath):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        if not os.path.isfile(figpath):
            raise FileNotFoundError("File does not exist")

        model = models.resnet50(pretrained=True)

        # 数据预处理步骤
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
        # stage1输出的特征值
        model1 = torch.nn.Sequential(*(list(model.children())[:-5]))
        model1.eval()
        # stage2输出的特征值
        model2 = torch.nn.Sequential(*(list(model.children())[:-4]))
        model2.eval()
        # stage3输出的特征值
        model3 = torch.nn.Sequential(*(list(model.children())[:-3]))
        model3.eval()
        # stage4输出的特征值
        model4 = torch.nn.Sequential(*(list(model.children())[:-2]))
        model4.eval()
        # 最后层输出的特征值
        model5 = torch.nn.Sequential(*(list(model.children())[:-1]))
        model5.eval()
        output = [0, 1, 2, 3, 4]
        # 提取特征
        with torch.no_grad():
            output[0] = model1(batch_t)
            output[1] = model2(batch_t)
            output[2] = model3(batch_t)
            output[3] = model4(batch_t)
            output[4] = model5(batch_t)
        """
        for i in range(5):
            print(output[i].size())
        """
        return output

    except FileNotFoundError as e:
        logging.error("Error: %s", e)
        return None
    except Exception as e:
        logging.error("An error occurred: %s", e)
        return None


# 以图像变量为输入
def extract_resnet18_content(image):
    """ 提取图像的ResNet18特征 """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 检查输入是否为PIL图像并转换为NumPy数组
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是PIL图像对象")

        # 加载预训练的 ResNet18 模型
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # 移除最后的全连接层，以提取特征而不进行分类
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

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


def extract_resnet50_content(image):
    """ 提取图像的ResNet50特征 """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 检查输入是否为PIL图像并转换为NumPy数组
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是PIL图像对象")

        # 加载预训练的 ResNet50 模型
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # 移除最后的全连接层，以提取特征而不进行分类
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

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


def extract_resnet152_content(image):
    """ 提取图像的ResNet152特征 """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # 检查输入是否为PIL图像并转换为NumPy数组
        if not isinstance(image, Image.Image):
            raise ValueError("输入必须是PIL图像对象")

        # 加载预训练的 ResNet152 模型
        model = resnet152(weights=ResNet152_Weights.DEFAULT)
        # 移除最后的全连接层，以提取特征而不进行分类
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

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
