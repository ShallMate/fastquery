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

import os
import sys
from extract.nets.resnet import *
from extract.nets.inception_v3 import *
from extract.nets.alexnet import *
from extract.nets.densenet import *
from extract.nets.googlenet import *
from extract.nets.mobilenet import *
from extract.nets.vggnet import *
import numpy as np
import torch.nn.functional as F
import torch
import pdb

# 计算各个网络的欧式距离1
# 计算resnet18网络下两张图片的欧式距离1


def compute_resnet18_Euclidean_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet18(figpath1)
    features2 = extract_resnet18(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算欧式距离
    resnet18_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - features2)))
    return resnet18_Euclidean_Distance

# 计算resnet50网络下两张图片的欧式距离2


def compute_resnet50_Euclidean_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet50(figpath1)
    features2 = extract_resnet50(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算欧式距离
    resnet50_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - features2)))
    return resnet50_Euclidean_Distance

# 计算resnet152网络下两张图片的欧式距离3


def compute_resnet152_Euclidean_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet152(figpath1)
    features2 = extract_resnet152(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算欧式距离
    resnet152_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - features2)))
    return resnet152_Euclidean_Distance

# 计算inception_v3网络下两张图片的欧式距离4


def compute_inception_v3_Euclidean_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_inception_v3(figpath1)
    features2 = extract_inception_v3(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算欧式距离
    inception_v3_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - features2)))
    return inception_v3_Euclidean_Distance

# 计算alexnet网络下两张图片的欧式距离5


def compute_alexnet_Euclidean_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_alexnet(figpath1)
    features2 = extract_alexnet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算欧式距离
    alexnet_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - features2)))
    return alexnet_Euclidean_Distance

# 计算densenet网络下两张图片的欧式距离6


def compute_densenet_Euclidean_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_densenet(figpath1)
    features2 = extract_densenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算欧式距离
    densenet_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - features2)))
    return densenet_Euclidean_Distance

# 计算googlenet网络下两张图片的欧式距离7


def compute_googlenet_Euclidean_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_googlenet(figpath1)
    features2 = extract_googlenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算欧式距离
    googlenet_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - features2)))
    return googlenet_Euclidean_Distance

# 计算mobilenet网络下两张图片的欧式距离8


def compute_mobilenet_Euclidean_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_mobilenet(figpath1)
    features2 = extract_mobilenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算欧式距离
    mobilenet_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - features2)))
    return mobilenet_Euclidean_Distance

# 计算vggnet网络下两张图片的欧式距离9


def compute_vggnet_Euclidean_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_vggnet(figpath1)
    features2 = extract_vggnet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算欧式距离
    vggnet_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - features2)))
    return vggnet_Euclidean_Distance

# 计算曼哈顿距离2
# 计算resnet18网络下两张图片的曼哈顿距离1


def compute_resnet18_Manhattan_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet18(figpath1)
    features2 = extract_resnet18(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
   # 第二步，计算曼哈顿距离
    resnet18_Manhattan_Distance = np.sum(np.abs(features1 - features2))
    return resnet18_Manhattan_Distance

# 计算resnet50网络下两张图片的曼哈顿距离2


def compute_resnet50_Manhattan_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet50(figpath1)
    features2 = extract_resnet50(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
   # 第二步，计算曼哈顿距离
    resnet50_Manhattan_Distance = np.sum(np.abs(features1 - features2))
    return resnet50_Manhattan_Distance

# 计算resnet152网络下两张图片的曼哈顿距离3


def compute_resnet152_Manhattan_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet152(figpath1)
    features2 = extract_resnet152(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
   # 第二步，计算曼哈顿距离
    resnet152_Manhattan_Distance = np.sum(np.abs(features1 - features2))
    return resnet152_Manhattan_Distance

# 计算inception_v3网络下两张图片的曼哈顿距离4


def compute_inception_v3_Manhattan_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_inception_v3(figpath1)
    features2 = extract_inception_v3(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
   # 第二步，计算曼哈顿距离
    inception_v3_Manhattan_Distance = np.sum(np.abs(features1 - features2))
    return inception_v3_Manhattan_Distance

# 计算alexnet网络下两张图片的曼哈顿距离5


def compute_alexnet_Manhattan_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_alexnet(figpath1)
    features2 = extract_alexnet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
   # 第二步，计算曼哈顿距离
    alexnet_Manhattan_Distance = np.sum(np.abs(features1 - features2))
    return alexnet_Manhattan_Distance

# 计算densenet网络下两张图片的曼哈顿距离6


def compute_densenet_Manhattan_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_densenet(figpath1)
    features2 = extract_densenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
   # 第二步，计算曼哈顿距离
    densenet_Manhattan_Distance = np.sum(np.abs(features1 - features2))
    return densenet_Manhattan_Distance

# 计算googlenet网络下两张图片的曼哈顿距离7


def compute_googlenet_Manhattan_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_googlenet(figpath1)
    features2 = extract_googlenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
   # 第二步，计算曼哈顿距离
    googlenet_Manhattan_Distance = np.sum(np.abs(features1 - features2))
    return googlenet_Manhattan_Distance

# 计算mobilenet网络下两张图片的曼哈顿距离8


def compute_mobilenet_Manhattan_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_mobilenet(figpath1)
    features2 = extract_mobilenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
   # 第二步，计算曼哈顿距离
    mobilenet_Manhattan_Distance = np.sum(np.abs(features1 - features2))
    return mobilenet_Manhattan_Distance

# 计算vggnet网络下两张图片的曼哈顿距离9


def compute_vggnet_Manhattan_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_vggnet(figpath1)
    features2 = extract_vggnet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
   # 第二步，计算曼哈顿距离
    vggnet_Manhattan_Distance = np.sum(np.abs(features1 - features2))
    return vggnet_Manhattan_Distance

# 计算切比雪夫距离3
# 计算resnet18网络下两张图片的切比雪夫距离1


def compute_resnet18_Chebyshev_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet18(figpath1)
    features2 = extract_resnet18(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算切比雪夫距离
    resnet18_Chebyshev_Distance = np.abs(features1 - features2).max()
    return resnet18_Chebyshev_Distance

# 计算resnet50网络下两张图片的切比雪夫距离2


def compute_resnet50_Chebyshev_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet50(figpath1)
    features2 = extract_resnet50(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算切比雪夫距离
    resnet50_Chebyshev_Distance = np.abs(features1 - features2).max()
    return resnet50_Chebyshev_Distance

# 计算resnet152网络下两张图片的切比雪夫距离3


def compute_resnet152_Chebyshev_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet152(figpath1)
    features2 = extract_resnet152(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算切比雪夫距离
    resnet152_Chebyshev_Distance = np.abs(features1 - features2).max()
    return resnet152_Chebyshev_Distance

# 计算inception_v3网络下两张图片的切比雪夫距离4


def compute_inception_v3_Chebyshev_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_inception_v3(figpath1)
    features2 = extract_inception_v3(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算切比雪夫距离
    inception_v3_Chebyshev_Distance = np.abs(features1 - features2).max()
    return inception_v3_Chebyshev_Distance

# 计算alexnet网络下两张图片的切比雪夫距离5


def compute_alexnet_Chebyshev_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_alexnet(figpath1)
    features2 = extract_alexnet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算切比雪夫距离
    alexnet_Chebyshev_Distance = np.abs(features1 - features2).max()
    return alexnet_Chebyshev_Distance

# 计算densenet网络下两张图片的切比雪夫距离6


def compute_densenet_Chebyshev_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_densenet(figpath1)
    features2 = extract_densenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算切比雪夫距离
    densenet_Chebyshev_Distance = np.abs(features1 - features2).max()
    return densenet_Chebyshev_Distance

# 计算googlenet网络下两张图片的切比雪夫距离7


def compute_googlenet_Chebyshev_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_googlenet(figpath1)
    features2 = extract_googlenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算切比雪夫距离
    googlenet_Chebyshev_Distance = np.abs(features1 - features2).max()
    return googlenet_Chebyshev_Distance

# 计算mobilenet网络下两张图片的切比雪夫距离8


def compute_mobilenet_Chebyshev_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_mobilenet(figpath1)
    features2 = extract_mobilenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算切比雪夫距离
    mobilenet_Chebyshev_Distance = np.abs(features1 - features2).max()
    return mobilenet_Chebyshev_Distance

# 计算vggnet网络下两张图片的切比雪夫距离9


def compute_vggnet_Chebyshev_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_vggnet(figpath1)
    features2 = extract_vggnet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算切比雪夫距离
    vggnet_Chebyshev_Distance = np.abs(features1 - features2).max()
    return vggnet_Chebyshev_Distance

# 计算余弦相似度4
# 求一个数组内所有元素的平均值


def compute_AVGarray(array):
    total = np.sum(array)
    # number是数组内元素的个数
    number = 0
    for row in array:
        number = number + len(row)
    # 平均值等于元素之和除以元素的个数
    avg = total / number
    return avg

# 计算resnet18网络下两张图片的余弦相似度1


def compute_resnet18_Cosine_Similarity(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet18(figpath1)
    features2 = extract_resnet18(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 删除冗余维度才能做dot运算
    features1 = features1.squeeze()
    features2 = features2.squeeze()
    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(features2)
    resnet18_Cosine_Similarity = np.dot(features1, features2) / temp
    return resnet18_Cosine_Similarity

# 计算resnet50网络下两张图片的余弦相似度2


def compute_resnet50_Cosine_Similarity(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet50(figpath1)
    features2 = extract_resnet50(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 删除冗余维度才能做dot运算
    features1 = features1.squeeze()
    features2 = features2.squeeze()
    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(features2)
    resnet50_Cosine_Similarity = np.dot(features1, features2) / temp
    return resnet50_Cosine_Similarity

# 计算resnet152网络下两张图片的余弦相似度3


def compute_resnet152_Cosine_Similarity(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_resnet152(figpath1)
    features2 = extract_resnet152(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 删除冗余维度才能做dot运算
    features1 = features1.squeeze()
    features2 = features2.squeeze()
    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(features2)
    resnet152_Cosine_Similarity = np.dot(features1, features2) / temp
    return resnet152_Cosine_Similarity

# 计算inception_v3网络下两张图片的余弦相似度4


def compute_inception_v3_Cosine_Similarity(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_inception_v3(figpath1)
    features2 = extract_inception_v3(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 删除冗余维度才能做dot运算
    features1 = features1.squeeze()
    features2 = features2.squeeze()
    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(features2)
    inception_v3_Cosine_Similarity = np.dot(features1, features2) / temp
    return inception_v3_Cosine_Similarity

# 计算alexnet网络下两张图片的余弦相似度5


def compute_alexnet_Cosine_Similarity(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_alexnet(figpath1)
    features2 = extract_alexnet(figpath2)
    # 第二步，计算余弦相似度
    features1_squeezed = features1.squeeze(0)
    features2_squeezed = features2.squeeze(0)
    alexnet_Cosine_Similarity = F.cosine_similarity(
        features1_squeezed, features2_squeezed, dim=0)
    alexnet_Cosine_Similarity = alexnet_Cosine_Similarity.numpy()
    alexnet_Cosine_Similarity = compute_AVGarray(alexnet_Cosine_Similarity)
    return alexnet_Cosine_Similarity

# 计算densenet网络下两张图片的余弦相似度6


def compute_densenet_Cosine_Similarity(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_densenet(figpath1)
    features2 = extract_densenet(figpath2)
    # 第二步，计算余弦相似度
    features1_squeezed = features1.squeeze(0)
    features2_squeezed = features2.squeeze(0)
    densenet_Cosine_Similarity = F.cosine_similarity(
        features1_squeezed, features2_squeezed, dim=0)
    densenet_Cosine_Similarity = densenet_Cosine_Similarity.numpy()
    densenet_Cosine_Similarity = compute_AVGarray(densenet_Cosine_Similarity)
    return densenet_Cosine_Similarity

# 计算googlenet网络下两张图片的余弦相似度7


def compute_googlenet_Cosine_Similarity(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_googlenet(figpath1)
    features2 = extract_googlenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 删除冗余维度才能做dot运算
    features1 = features1.squeeze()
    features2 = features2.squeeze()
    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(features2)
    googlenet_Cosine_Similarity = np.dot(features1, features2) / temp
    return googlenet_Cosine_Similarity

# 计算mobilenet网络下两张图片的余弦相似度8


def compute_mobilenet_Cosine_Similarity(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_mobilenet(figpath1)
    features2 = extract_mobilenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 删除冗余维度才能做dot运算
    features1 = features1.squeeze()
    features2 = features2.squeeze()
    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(features2)
    mobilenet_Cosine_Similarity = np.dot(features1, features2) / temp
    return mobilenet_Cosine_Similarity

# 计算vggnet网络下两张图片的余弦相似度9


def compute_vggnet_Cosine_Similarity(figpath1, figpath2):
    # 第一步，提取特征值
    features1 = extract_vggnet(figpath1)
    features2 = extract_vggnet(figpath2)
    # 第二步，计算余弦相似度
    features1_squeezed = features1.squeeze(0)
    features2_squeezed = features2.squeeze(0)
    vggnet_Cosine_Similarity = F.cosine_similarity(
        features1_squeezed, features2_squeezed, dim=0)
    vggnet_Cosine_Similarity = vggnet_Cosine_Similarity.numpy()
    vggnet_Cosine_Similarity = compute_AVGarray(vggnet_Cosine_Similarity)
    return vggnet_Cosine_Similarity

# 计算闵可夫斯基距离5
# 计算resnet18网络下两张图片的闵可夫斯基距离1


def compute_resnet18_Minkowski_Distance(figpath1, figpath2, num):
    # 第一步，提取特征值
    features1 = extract_resnet18(figpath1)
    features2 = extract_resnet18(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算闵可夫斯基距离
    resnet18_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - features2, num)), 1 / num)
    return resnet18_Minkowski_Distance

# 计算resnet50网络下两张图片的闵可夫斯基距离2


def compute_resnet50_Minkowski_Distance(figpath1, figpath2, num):
    # 第一步，提取特征值
    features1 = extract_resnet50(figpath1)
    features2 = extract_resnet50(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算闵可夫斯基距离
    resnet50_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - features2, num)), 1 / num)
    return resnet50_Minkowski_Distance

# 计算resnet152网络下两张图片的闵可夫斯基距离3


def compute_resnet152_Minkowski_Distance(figpath1, figpath2, num):
    # 第一步，提取特征值
    features1 = extract_resnet152(figpath1)
    features2 = extract_resnet152(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算闵可夫斯基距离
    resnet152_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - features2, num)), 1 / num)
    return resnet152_Minkowski_Distance

# 计算inception_v3网络下两张图片的闵可夫斯基距离4


def compute_inception_v3_Minkowski_Distance(figpath1, figpath2, num):
    # 第一步，提取特征值
    features1 = extract_inception_v3(figpath1)
    features2 = extract_inception_v3(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算闵可夫斯基距离
    inception_v3_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - features2, num)), 1 / num)
    return inception_v3_Minkowski_Distance

# 计算alexnet网络下两张图片的闵可夫斯基距离5


def compute_alexnet_Minkowski_Distance(figpath1, figpath2, num):
    # 第一步，提取特征值
    features1 = extract_alexnet(figpath1)
    features2 = extract_alexnet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算闵可夫斯基距离
    alexnet_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - features2, num)), 1 / num)
    return alexnet_Minkowski_Distance

# 计算densenet网络下两张图片的闵可夫斯基距离6


def compute_densenet_Minkowski_Distance(figpath1, figpath2, num):
    # 第一步，提取特征值
    features1 = extract_densenet(figpath1)
    features2 = extract_densenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算闵可夫斯基距离
    densenet_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - features2, num)), 1 / num)
    return densenet_Minkowski_Distance

# 计算googlenet网络下两张图片的闵可夫斯基距离7


def compute_googlenet_Minkowski_Distance(figpath1, figpath2, num):
    # 第一步，提取特征值
    features1 = extract_googlenet(figpath1)
    features2 = extract_googlenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算闵可夫斯基距离
    googlenet_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - features2, num)), 1 / num)
    return googlenet_Minkowski_Distance

# 计算mobilenet网络下两张图片的闵可夫斯基距离8


def compute_mobilenet_Minkowski_Distance(figpath1, figpath2, num):
    # 第一步，提取特征值
    features1 = extract_mobilenet(figpath1)
    features2 = extract_mobilenet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算闵可夫斯基距离
    mobilenet_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - features2, num)), 1 / num)
    return mobilenet_Minkowski_Distance

# 计算vggnet网络下两张图片的闵可夫斯基距离9


def compute_vggnet_Minkowski_Distance(figpath1, figpath2, num):
    # 第一步，提取特征值
    features1 = extract_vggnet(figpath1)
    features2 = extract_vggnet(figpath2)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 第二步，计算闵可夫斯基距离
    vggnet_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - features2, num)), 1 / num)
    return vggnet_Minkowski_Distance

# 计算杰卡德距离6
# 计算resnet18网络下两张图片的杰卡德距离1


def compute_resnet18_Jaccard_Distance(figpath1, figpath2, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    features1 = extract_resnet18(figpath1)
    features2 = extract_resnet18(figpath2)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    resnet18_Jaccard_Distance = 1.0 - intersection / union
    resnet18_Jaccard_Distance = resnet18_Jaccard_Distance.numpy()
    return resnet18_Jaccard_Distance

# 计算resnet50网络下两张图片的杰卡德距离2


def compute_resnet50_Jaccard_Distance(figpath1, figpath2, threshold):
    # 第一步，提取特征值
    features1 = extract_resnet50(figpath1)
    features2 = extract_resnet50(figpath2)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    resnet50_Jaccard_Distance = 1.0 - intersection / union
    resnet50_Jaccard_Distance = resnet50_Jaccard_Distance.numpy()
    return resnet50_Jaccard_Distance

# 计算resnet152网络下两张图片的杰卡德距离3


def compute_resnet152_Jaccard_Distance(figpath1, figpath2, threshold):
    # 第一步，提取特征值
    features1 = extract_resnet152(figpath1)
    features2 = extract_resnet152(figpath2)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    resnet152_Jaccard_Distance = 1.0 - intersection / union
    resnet152_Jaccard_Distance = resnet152_Jaccard_Distance.numpy()
    return resnet152_Jaccard_Distance

# 计算inception_v3网络下两张图片的杰卡德距离4


def compute_inception_v3_Jaccard_Distance(figpath1, figpath2, threshold):
    # 第一步，提取特征值
    features1 = extract_inception_v3(figpath1)
    features2 = extract_inception_v3(figpath2)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    inception_v3_Jaccard_Distance = 1.0 - intersection / union
    inception_v3_Jaccard_Distance = inception_v3_Jaccard_Distance.numpy()
    return inception_v3_Jaccard_Distance

# 计算alexnet网络下两张图片的杰卡德距离5


def compute_alexnet_Jaccard_Distance(figpath1, figpath2, threshold):
    # 第一步，提取特征值
    features1 = extract_alexnet(figpath1)
    features2 = extract_alexnet(figpath2)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    alexnet_Jaccard_Distance = 1.0 - intersection / union
    alexnet_Jaccard_Distance = alexnet_Jaccard_Distance.numpy()
    return alexnet_Jaccard_Distance

# 计算densenet网络下两张图片的杰卡德距离6


def compute_densenet_Jaccard_Distance(figpath1, figpath2, threshold):
    # 第一步，提取特征值
    features1 = extract_densenet(figpath1)
    features2 = extract_densenet(figpath2)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    densenet_Jaccard_Distance = 1.0 - intersection / union
    densenet_Jaccard_Distance = densenet_Jaccard_Distance.numpy()
    return densenet_Jaccard_Distance

# 计算googlenet网络下两张图片的杰卡德距离7


def compute_googlenet_Jaccard_Distance(figpath1, figpath2, threshold):
    # 第一步，提取特征值
    features1 = extract_googlenet(figpath1)
    features2 = extract_googlenet(figpath2)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    googlenet_Jaccard_Distance = 1.0 - intersection / union
    googlenet_Jaccard_Distance = googlenet_Jaccard_Distance.numpy()
    return googlenet_Jaccard_Distance

# 计算mobilenet网络下两张图片的杰卡德距离8


def compute_mobilenet_Jaccard_Distance(figpath1, figpath2, threshold):
    # 第一步，提取特征值
    features1 = extract_mobilenet(figpath1)
    features2 = extract_mobilenet(figpath2)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    mobilenet_Jaccard_Distance = 1.0 - intersection / union
    mobilenet_Jaccard_Distance = mobilenet_Jaccard_Distance.numpy()
    return mobilenet_Jaccard_Distance

# 计算vggnet网络下两张图片的杰卡德距离9


def compute_vggnet_Jaccard_Distance(figpath1, figpath2, threshold):
    # 第一步，提取特征值
    features1 = extract_vggnet(figpath1)
    features2 = extract_vggnet(figpath2)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    vggnet_Jaccard_Distance = 1.0 - intersection / union
    vggnet_Jaccard_Distance = vggnet_Jaccard_Distance.numpy()
    return vggnet_Jaccard_Distance


# 输入为一个tensor向量，输出一个同结构的，但是对应位置元素均替换为1的tensor向量
def replace_with_ones_and_keep_as_tensor(data):
    if data.dim() == 0:
        return torch.tensor(1)
    else:
        return data.new_full(size=data.size(), fill_value=1)

# 计算欧式距离1
# 计算resnet18网络下某张图片和一向量的欧式距离1


def compute_resnet18_with1_Euclidean_Distance(figpath):

    # 第一步，提取特征值
    features1 = extract_resnet18(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    resnet18_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return resnet18_with1_Euclidean_Distance

# 计算resnet50网络下某张图片和一向量的欧式距离2


def compute_resnet50_with1_Euclidean_Distance(figpath):

    # 第一步，提取特征值
    features1 = extract_resnet50(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    resnet50_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return resnet50_with1_Euclidean_Distance

# 计算resnet152网络下某张图片和一向量的欧式距离3


def compute_resnet152_with1_Euclidean_Distance(figpath):

    # 第一步，提取特征值
    features1 = extract_resnet152(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    resnet152_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return resnet152_with1_Euclidean_Distance

# 计算inception_v3网络下某张图片和一向量的欧式距离4


def compute_inception_v3_with1_Euclidean_Distance(figpath):

    # 第一步，提取特征值
    features1 = extract_inception_v3(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    inception_v3_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return inception_v3_with1_Euclidean_Distance

# 计算alexnet网络下某张图片和一向量的欧式距离5


def compute_alexnet_with1_Euclidean_Distance(figpath):

    # 第一步，提取特征值
    features1 = extract_alexnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    alexnet_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return alexnet_with1_Euclidean_Distance

# 计算densenet网络下某张图片和一向量的欧式距离6


def compute_densenet_with1_Euclidean_Distance(figpath):

    # 第一步，提取特征值
    features1 = extract_densenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    densenet_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return densenet_with1_Euclidean_Distance

# 计算googlenet网络下某张图片和一向量的欧式距离7


def compute_googlenet_with1_Euclidean_Distance(figpath):

    # 第一步，提取特征值
    features1 = extract_googlenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    googlenet_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return googlenet_with1_Euclidean_Distance

# 计算mobilenet网络下某张图片和一向量的欧式距离8


def compute_mobilenet_with1_Euclidean_Distance(figpath):

    # 第一步，提取特征值
    features1 = extract_mobilenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    mobilenet_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return mobilenet_with1_Euclidean_Distance

# 计算vggnet网络下某张图片和一向量的欧式距离9


def compute_vggnet_with1_Euclidean_Distance(figpath):

    # 第一步，提取特征值
    features1 = extract_vggnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    vggnet_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return vggnet_with1_Euclidean_Distance

# 计算曼哈顿距离2
# 计算resnet18网络下某张图片和一向量的曼哈顿距离1


def compute_resnet18_with1_Manhattan_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_resnet18(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    resnet18_with1_Manhattan_Distance = np.sum(np.abs(features1 - replace_one))
    return resnet18_with1_Manhattan_Distance

# 计算resnet50网络下某张图片和一向量的曼哈顿距离2


def compute_resnet50_with1_Manhattan_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_resnet50(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    resnet50_with1_Manhattan_Distance = np.sum(np.abs(features1 - replace_one))
    return resnet50_with1_Manhattan_Distance

# 计算resnet152网络下某张图片和一向量的曼哈顿距离3


def compute_resnet152_with1_Manhattan_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_resnet152(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    resnet152_with1_Manhattan_Distance = np.sum(
        np.abs(features1 - replace_one))
    return resnet152_with1_Manhattan_Distance

# 计算inception_v3网络下某张图片和一向量的曼哈顿距离4


def compute_inception_v3_with1_Manhattan_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_inception_v3(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    inception_v3_with1_Manhattan_Distance = np.sum(
        np.abs(features1 - replace_one))
    return inception_v3_with1_Manhattan_Distance

# 计算alexnet网络下某张图片和一向量的曼哈顿距离5


def compute_alexnet_with1_Manhattan_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_alexnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    alexnet_with1_Manhattan_Distance = np.sum(np.abs(features1 - replace_one))
    return alexnet_with1_Manhattan_Distance

# 计算densenet网络下某张图片和一向量的曼哈顿距离6


def compute_densenet_with1_Manhattan_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_densenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    densenet_with1_Manhattan_Distance = np.sum(np.abs(features1 - replace_one))
    return densenet_with1_Manhattan_Distance

# 计算googlenet网络下某张图片和一向量的曼哈顿距离7


def compute_googlenet_with1_Manhattan_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_googlenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    googlenet_with1_Manhattan_Distance = np.sum(
        np.abs(features1 - replace_one))
    return googlenet_with1_Manhattan_Distance

# 计算mobilenet网络下某张图片和一向量的曼哈顿距离8


def compute_mobilenet_with1_Manhattan_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_mobilenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    mobilenet_with1_Manhattan_Distance = np.sum(
        np.abs(features1 - replace_one))
    return mobilenet_with1_Manhattan_Distance

# 计算vggnet网络下某张图片和一向量的曼哈顿距离9


def compute_vggnet_with1_Manhattan_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_vggnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    vggnet_with1_Manhattan_Distance = np.sum(np.abs(features1 - replace_one))
    return vggnet_with1_Manhattan_Distance

# 计算切比雪夫距离3
# 计算resnet18网络下某张图片和一向量的切比雪夫距离1


def compute_resnet18_with1_Chebyshev_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_resnet18(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    resnet18_with1_Chebyshev_Distance = np.abs(features1 - replace_one).max()
    return resnet18_with1_Chebyshev_Distance

# 计算resnet50网络下某张图片和一向量的切比雪夫距离2


def compute_resnet50_with1_Chebyshev_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_resnet50(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    resnet50_with1_Chebyshev_Distance = np.abs(features1 - replace_one).max()
    return resnet50_with1_Chebyshev_Distance

# 计算resnet152网络下某张图片和一向量的切比雪夫距离3


def compute_resnet152_with1_Chebyshev_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_resnet152(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    resnet152_with1_Chebyshev_Distance = np.abs(features1 - replace_one).max()
    return resnet152_with1_Chebyshev_Distance

# 计算inception_v3网络下某张图片和一向量的切比雪夫距离4


def compute_inception_v3_with1_Chebyshev_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_inception_v3(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    inception_v3_with1_Chebyshev_Distance = np.abs(
        features1 - replace_one).max()
    return inception_v3_with1_Chebyshev_Distance

# 计算alexnet网络下某张图片和一向量的切比雪夫距离5


def compute_alexnet_with1_Chebyshev_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_alexnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    alexnet_with1_Chebyshev_Distance = np.abs(features1 - replace_one).max()
    return alexnet_with1_Chebyshev_Distance

# 计算densenet网络下某张图片和一向量的切比雪夫距离6


def compute_densenet_with1_Chebyshev_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_densenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    densenet_with1_Chebyshev_Distance = np.abs(features1 - replace_one).max()
    return densenet_with1_Chebyshev_Distance

# 计算googlenet网络下某张图片和一向量的切比雪夫距离7


def compute_googlenet_with1_Chebyshev_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_googlenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    googlenet_with1_Chebyshev_Distance = np.abs(features1 - replace_one).max()
    return googlenet_with1_Chebyshev_Distance

# 计算mobilenet网络下某张图片和一向量的切比雪夫距离8


def compute_mobilenet_with1_Chebyshev_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_mobilenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    mobilenet_with1_Chebyshev_Distance = np.abs(features1 - replace_one).max()
    return mobilenet_with1_Chebyshev_Distance

# 计算vggnet网络下某张图片和一向量的切比雪夫距离9


def compute_vggnet_with1_Chebyshev_Distance(figpath):
    # 第一步，提取特征值
    features1 = extract_vggnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    vggnet_with1_Chebyshev_Distance = np.abs(features1 - replace_one).max()
    return vggnet_with1_Chebyshev_Distance

# 计算余弦相似度4
# 计算resnet18网络下某张图片和一向量的余弦相似度1


def compute_resnet18_with1_Cosine_Similarity(figpath):

    # 提取图片特征值
    features1 = extract_resnet18(figpath)
    # 将张量对应位置的元素全部换为1
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 变为array类型
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 删除冗余维度
    features1 = features1.squeeze()
    replace_one = replace_one.squeeze()

    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(replace_one)
    resnet18_with1_Cosine_Similarity = np.dot(features1, replace_one) / temp

    return resnet18_with1_Cosine_Similarity

# 计算resnet50网络下某张图片和一向量的余弦相似度2


def compute_resnet50_with1_Cosine_Similarity(figpath):

    # 提取图片特征值
    features1 = extract_resnet50(figpath)
    # 将张量对应位置的元素全部换为1
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 变为array类型
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 删除冗余维度
    features1 = features1.squeeze()
    replace_one = replace_one.squeeze()

    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(replace_one)
    resnet50_with1_Cosine_Similarity = np.dot(features1, replace_one) / temp

    return resnet50_with1_Cosine_Similarity

# 计算resnet152网络下某张图片和一向量的余弦相似度3


def compute_resnet152_with1_Cosine_Similarity(figpath):

    # 提取图片特征值
    features1 = extract_resnet152(figpath)
    # 将张量对应位置的元素全部换为1
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 变为array类型
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 删除冗余维度
    features1 = features1.squeeze()
    replace_one = replace_one.squeeze()

    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(replace_one)
    resnet152_with1_Cosine_Similarity = np.dot(features1, replace_one) / temp

    return resnet152_with1_Cosine_Similarity

# 计算inception_v3网络下某张图片和一向量的余弦相似度4


def compute_inception_v3_with1_Cosine_Similarity(figpath):

    # 提取图片特征值
    features1 = extract_inception_v3(figpath)
    # 将张量对应位置的元素全部换为1
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 变为array类型
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 删除冗余维度
    features1 = features1.squeeze()
    replace_one = replace_one.squeeze()

    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(replace_one)
    inception_v3_with1_Cosine_Similarity = np.dot(
        features1, replace_one) / temp

    return inception_v3_with1_Cosine_Similarity

# 计算alexnet网络下某张图片和一向量的余弦相似度5


def compute_alexnet_with1_Cosine_Similarity(figpath):

    # 第一步，提取特征值
    features1 = extract_alexnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 第二步，计算余弦相似度
    features1_squeezed = features1.squeeze(0)
    replace_one_squeezed = replace_one.squeeze(0)
    alexnet_with1_Cosine_Similarity = F.cosine_similarity(
        features1_squeezed, replace_one_squeezed, dim=0)
    alexnet_with1_Cosine_Similarity = alexnet_with1_Cosine_Similarity.numpy()
    alexnet_with1_Cosine_Similarity = compute_AVGarray(
        alexnet_with1_Cosine_Similarity)
    return alexnet_with1_Cosine_Similarity

# 计算densenet网络下某张图片和一向量的余弦相似度6


def compute_densenet_with1_Cosine_Similarity(figpath):

    # 第一步，提取特征值
    features1 = extract_densenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 第二步，计算余弦相似度
    features1_squeezed = features1.squeeze(0)
    replace_one_squeezed = replace_one.squeeze(0)
    densenet_with1_Cosine_Similarity = F.cosine_similarity(
        features1_squeezed, replace_one_squeezed, dim=0)
    densenet_with1_Cosine_Similarity = densenet_with1_Cosine_Similarity.numpy()
    densenet_with1_Cosine_Similarity = compute_AVGarray(
        densenet_with1_Cosine_Similarity)
    return densenet_with1_Cosine_Similarity

# 计算googlenet网络下某张图片和一向量的余弦相似度7


def compute_googlenet_with1_Cosine_Similarity(figpath):

    # 提取图片特征值
    features1 = extract_googlenet(figpath)
    # 将张量对应位置的元素全部换为1
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 变为array类型
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 删除冗余维度
    features1 = features1.squeeze()
    replace_one = replace_one.squeeze()

    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(replace_one)
    googlenet_with1_Cosine_Similarity = np.dot(features1, replace_one) / temp

    return googlenet_with1_Cosine_Similarity

# 计算mobilenet网络下某张图片和一向量的余弦相似度8


def compute_mobilenet_with1_Cosine_Similarity(figpath):

    # 提取图片特征值
    features1 = extract_mobilenet(figpath)
    # 将张量对应位置的元素全部换为1
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 变为array类型
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 删除冗余维度
    features1 = features1.squeeze()
    replace_one = replace_one.squeeze()

    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(replace_one)
    mobilenet_with1_Cosine_Similarity = np.dot(features1, replace_one) / temp

    return mobilenet_with1_Cosine_Similarity

# 计算vggnet网络下某张图片和一向量的余弦相似度9


def compute_vggnet_with1_Cosine_Similarity(figpath):

    # 第一步，提取特征值
    features1 = extract_vggnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 第二步，计算余弦相似度
    features1_squeezed = features1.squeeze(0)
    replace_one_squeezed = replace_one.squeeze(0)
    vggnet_with1_Cosine_Similarity = F.cosine_similarity(
        features1_squeezed, replace_one_squeezed, dim=0)
    vggnet_with1_Cosine_Similarity = vggnet_with1_Cosine_Similarity.numpy()
    vggnet_with1_Cosine_Similarity = compute_AVGarray(
        vggnet_with1_Cosine_Similarity)
    return vggnet_with1_Cosine_Similarity

# 计算闵可夫斯基距离5
# 计算resnet18网络下某张图片和一向量的闵可夫斯基距离1


def compute_resnet18_with1_Minkowski_Distance(figpath, num):
    # 第一步，提取特征值
    features1 = extract_resnet18(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算闵可夫斯基距离
    resnet18_with1_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - replace_one, num)), 1 / num)
    return resnet18_with1_Minkowski_Distance

# 计算resnet50网络下某张图片和一向量的闵可夫斯基距离2


def compute_resnet50_with1_Minkowski_Distance(figpath, num):
    # 第一步，提取特征值
    features1 = extract_resnet50(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算闵可夫斯基距离
    resnet50_with1_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - replace_one, num)), 1 / num)
    return resnet50_with1_Minkowski_Distance

# 计算resnet152网络下某张图片和一向量的闵可夫斯基距离3


def compute_resnet152_with1_Minkowski_Distance(figpath, num):
    # 第一步，提取特征值
    features1 = extract_resnet152(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算闵可夫斯基距离
    resnet152_with1_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - replace_one, num)), 1 / num)
    return resnet152_with1_Minkowski_Distance

# 计算inception_v3网络下某张图片和一向量的闵可夫斯基距离4


def compute_inception_v3_with1_Minkowski_Distance(figpath, num):
    # 第一步，提取特征值
    features1 = extract_inception_v3(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算闵可夫斯基距离
    inception_v3_with1_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - replace_one, num)), 1 / num)
    return inception_v3_with1_Minkowski_Distance

# 计算alexnet网络下某张图片和一向量的闵可夫斯基距离5


def compute_alexnet_with1_Minkowski_Distance(figpath, num):
    # 第一步，提取特征值
    features1 = extract_alexnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算闵可夫斯基距离
    alexnet_with1_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - replace_one, num)), 1 / num)
    return alexnet_with1_Minkowski_Distance

# 计算densenet网络下某张图片和一向量的闵可夫斯基距离6


def compute_densenet_with1_Minkowski_Distance(figpath, num):
    # 第一步，提取特征值
    features1 = extract_densenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算闵可夫斯基距离
    densenet_with1_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - replace_one, num)), 1 / num)
    return densenet_with1_Minkowski_Distance

# 计算googlenet网络下某张图片和一向量的闵可夫斯基距离7


def compute_googlenet_with1_Minkowski_Distance(figpath, num):
    # 第一步，提取特征值
    features1 = extract_googlenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算闵可夫斯基距离
    googlenet_with1_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - replace_one, num)), 1 / num)
    return googlenet_with1_Minkowski_Distance

# 计算mobilenet网络下某张图片和一向量的闵可夫斯基距离8


def compute_mobilenet_with1_Minkowski_Distance(figpath, num):
    # 第一步，提取特征值
    features1 = extract_mobilenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算闵可夫斯基距离
    mobilenet_with1_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - replace_one, num)), 1 / num)
    return mobilenet_with1_Minkowski_Distance

# 计算vggnet网络下某张图片和一向量的闵可夫斯基距离9


def compute_vggnet_with1_Minkowski_Distance(figpath, num):
    # 第一步，提取特征值
    features1 = extract_vggnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算闵可夫斯基距离
    vggnet_with1_Minkowski_Distance = np.power(
        np.sum(np.power(features1 - replace_one, num)), 1 / num)
    return vggnet_with1_Minkowski_Distance

# 计算杰卡德距离6
# 计算resnet18网络下某张图片和一向量的杰卡德距离1


def compute_resnet18_with1_Jaccard_Distance(figpath, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    features1 = extract_resnet18(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    replace_one_binary = torch.where(
        replace_one > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, replace_one_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, replace_one_binary))
    resnet18_with1_Jaccard_Distance = 1.0 - intersection / union
    resnet18_with1_Jaccard_Distance = resnet18_with1_Jaccard_Distance.numpy()
    return resnet18_with1_Jaccard_Distance

# 计算resnet50网络下某张图片和一向量的杰卡德距离2


def compute_resnet50_with1_Jaccard_Distance(figpath, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    features1 = extract_resnet50(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    replace_one_binary = torch.where(
        replace_one > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, replace_one_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, replace_one_binary))
    resnet50_with1_Jaccard_Distance = 1.0 - intersection / union
    resnet50_with1_Jaccard_Distance = resnet50_with1_Jaccard_Distance.numpy()
    return resnet50_with1_Jaccard_Distance

# 计算resnet152网络下某张图片和一向量的杰卡德距离3


def compute_resnet152_with1_Jaccard_Distance(figpath, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    features1 = extract_resnet152(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    replace_one_binary = torch.where(
        replace_one > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, replace_one_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, replace_one_binary))
    resnet152_with1_Jaccard_Distance = 1.0 - intersection / union
    resnet152_with1_Jaccard_Distance = resnet152_with1_Jaccard_Distance.numpy()
    return resnet152_with1_Jaccard_Distance

# 计算inception_v3网络下某张图片和一向量的杰卡德距离4


def compute_inception_v3_with1_Jaccard_Distance(figpath, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    features1 = extract_inception_v3(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    replace_one_binary = torch.where(
        replace_one > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, replace_one_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, replace_one_binary))
    inception_v3_with1_Jaccard_Distance = 1.0 - intersection / union
    inception_v3_with1_Jaccard_Distance = inception_v3_with1_Jaccard_Distance.numpy()
    return inception_v3_with1_Jaccard_Distance

# 计算alexnet网络下某张图片和一向量的杰卡德距离5


def compute_alexnet_with1_Jaccard_Distance(figpath, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    features1 = extract_alexnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    replace_one_binary = torch.where(
        replace_one > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, replace_one_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, replace_one_binary))
    alexnet_with1_Jaccard_Distance = 1.0 - intersection / union
    alexnet_with1_Jaccard_Distance = alexnet_with1_Jaccard_Distance.numpy()
    return alexnet_with1_Jaccard_Distance

# 计算densenet网络下某张图片和一向量的杰卡德距离6


def compute_densenet_with1_Jaccard_Distance(figpath, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    features1 = extract_densenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    replace_one_binary = torch.where(
        replace_one > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, replace_one_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, replace_one_binary))
    densenet_with1_Jaccard_Distance = 1.0 - intersection / union
    densenet_with1_Jaccard_Distance = densenet_with1_Jaccard_Distance.numpy()
    return densenet_with1_Jaccard_Distance

# 计算googlenet网络下某张图片和一向量的杰卡德距离7


def compute_googlenet_with1_Jaccard_Distance(figpath, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    features1 = extract_googlenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    replace_one_binary = torch.where(
        replace_one > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, replace_one_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, replace_one_binary))
    googlenet_with1_Jaccard_Distance = 1.0 - intersection / union
    googlenet_with1_Jaccard_Distance = googlenet_with1_Jaccard_Distance.numpy()
    return googlenet_with1_Jaccard_Distance

# 计算mobilenet网络下某张图片和一向量的杰卡德距离8


def compute_mobilenet_with1_Jaccard_Distance(figpath, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    features1 = extract_mobilenet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    replace_one_binary = torch.where(
        replace_one > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, replace_one_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, replace_one_binary))
    mobilenet_with1_Jaccard_Distance = 1.0 - intersection / union
    mobilenet_with1_Jaccard_Distance = mobilenet_with1_Jaccard_Distance.numpy()
    return mobilenet_with1_Jaccard_Distance

# 计算vggnet网络下某张图片和一向量的杰卡德距离9


def compute_vggnet_with1_Jaccard_Distance(figpath, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    features1 = extract_vggnet(figpath)
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    replace_one_binary = torch.where(
        replace_one > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, replace_one_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, replace_one_binary))
    vggnet_with1_Jaccard_Distance = 1.0 - intersection / union
    vggnet_with1_Jaccard_Distance = vggnet_with1_Jaccard_Distance.numpy()
    return vggnet_with1_Jaccard_Distance



# resnet50_specific_layers，两张图片，每层特征值对应的特征距离
# 计算resnet50__specific_layers网络下两张图片的欧式距离1


def compute_resnet50__specific_layers_Euclidean_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    feature1 = extract_resnet50_specific_layers(figpath1)
    feature11 = feature1[0]
    feature12 = feature1[1]
    feature13 = feature1[2]
    feature14 = feature1[3]
    feature15 = feature1[4]

    feature2 = extract_resnet50_specific_layers(figpath2)
    feature21 = feature2[0]
    feature22 = feature2[1]
    feature23 = feature2[2]
    feature24 = feature2[3]
    feature25 = feature2[4]

    feature11 = feature11.numpy()
    feature12 = feature12.numpy()
    feature13 = feature13.numpy()
    feature14 = feature14.numpy()
    feature15 = feature15.numpy()

    feature21 = feature21.numpy()
    feature22 = feature22.numpy()
    feature23 = feature23.numpy()
    feature24 = feature24.numpy()
    feature25 = feature25.numpy()

    # 第二步，计算欧式距离
    resnet50__specific_layers_Euclidean_Distance = [1, 2, 3, 4, 5]
    resnet50__specific_layers_Euclidean_Distance[0] = np.sqrt(
        np.sum(np.square(feature11 - feature21)))
    resnet50__specific_layers_Euclidean_Distance[1] = np.sqrt(
        np.sum(np.square(feature12 - feature22)))
    resnet50__specific_layers_Euclidean_Distance[2] = np.sqrt(
        np.sum(np.square(feature13 - feature23)))
    resnet50__specific_layers_Euclidean_Distance[3] = np.sqrt(
        np.sum(np.square(feature14 - feature24)))
    resnet50__specific_layers_Euclidean_Distance[4] = np.sqrt(
        np.sum(np.square(feature15 - feature25)))

    return resnet50__specific_layers_Euclidean_Distance

# 计算resnet50__specific_layers网络下两张图片的曼哈顿距离2


def compute_resnet50__specific_layers_Manhattan_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    feature1 = extract_resnet50_specific_layers(figpath1)
    feature11 = feature1[0]
    feature12 = feature1[1]
    feature13 = feature1[2]
    feature14 = feature1[3]
    feature15 = feature1[4]

    feature2 = extract_resnet50_specific_layers(figpath2)
    feature21 = feature2[0]
    feature22 = feature2[1]
    feature23 = feature2[2]
    feature24 = feature2[3]
    feature25 = feature2[4]

    feature11 = feature11.numpy()
    feature12 = feature12.numpy()
    feature13 = feature13.numpy()
    feature14 = feature14.numpy()
    feature15 = feature15.numpy()

    feature21 = feature21.numpy()
    feature22 = feature22.numpy()
    feature23 = feature23.numpy()
    feature24 = feature24.numpy()
    feature25 = feature25.numpy()

    # 第二步，计算曼哈顿距离
    resnet50__specific_layers_Manhattan_Distance = [1, 2, 3, 4, 5]
    resnet50__specific_layers_Manhattan_Distance[0] = np.sum(
        np.abs(feature11 - feature21))
    resnet50__specific_layers_Manhattan_Distance[1] = np.sum(
        np.abs(feature12 - feature22))
    resnet50__specific_layers_Manhattan_Distance[2] = np.sum(
        np.abs(feature13 - feature23))
    resnet50__specific_layers_Manhattan_Distance[3] = np.sum(
        np.abs(feature14 - feature24))
    resnet50__specific_layers_Manhattan_Distance[4] = np.sum(
        np.abs(feature15 - feature25))

    return resnet50__specific_layers_Manhattan_Distance

# 计算resnet50__specific_layers网络下两张图片的切比雪夫距离3


def compute_resnet50__specific_layers_Chebyshev_Distance(figpath1, figpath2):
    # 第一步，提取特征值
    feature1 = extract_resnet50_specific_layers(figpath1)
    feature11 = feature1[0]
    feature12 = feature1[1]
    feature13 = feature1[2]
    feature14 = feature1[3]
    feature15 = feature1[4]

    feature2 = extract_resnet50_specific_layers(figpath2)
    feature21 = feature2[0]
    feature22 = feature2[1]
    feature23 = feature2[2]
    feature24 = feature2[3]
    feature25 = feature2[4]

    feature11 = feature11.numpy()
    feature12 = feature12.numpy()
    feature13 = feature13.numpy()
    feature14 = feature14.numpy()
    feature15 = feature15.numpy()

    feature21 = feature21.numpy()
    feature22 = feature22.numpy()
    feature23 = feature23.numpy()
    feature24 = feature24.numpy()
    feature25 = feature25.numpy()

    # 第二步，计算切比雪夫距离
    resnet50__specific_layers_Chebyshev_Distance = [1, 2, 3, 4, 5]
    resnet50__specific_layers_Chebyshev_Distance[0] = np.abs(
        feature11 - feature21).max()
    resnet50__specific_layers_Chebyshev_Distance[1] = np.abs(
        feature12 - feature22).max()
    resnet50__specific_layers_Chebyshev_Distance[2] = np.abs(
        feature13 - feature23).max()
    resnet50__specific_layers_Chebyshev_Distance[3] = np.abs(
        feature14 - feature24).max()
    resnet50__specific_layers_Chebyshev_Distance[4] = np.abs(
        feature15 - feature25).max()

    return resnet50__specific_layers_Chebyshev_Distance

# 计算resnet50__specific_layers网络下两张图片的余弦相似度4


def compute_resnet50__specific_layers_Cosine_Similarity(figpath1, figpath2):
    # 第一步，提取特征值
    feature1 = extract_resnet50_specific_layers(figpath1)
    feature11 = feature1[0]
    feature12 = feature1[1]
    feature13 = feature1[2]
    feature14 = feature1[3]
    feature15 = feature1[4]

    feature2 = extract_resnet50_specific_layers(figpath2)
    feature21 = feature2[0]
    feature22 = feature2[1]
    feature23 = feature2[2]
    feature24 = feature2[3]
    feature25 = feature2[4]

    feature11 = feature11.numpy()
    feature12 = feature12.numpy()
    feature13 = feature13.numpy()
    feature14 = feature14.numpy()
    feature15 = feature15.numpy()

    feature21 = feature21.numpy()
    feature22 = feature22.numpy()
    feature23 = feature23.numpy()
    feature24 = feature24.numpy()
    feature25 = feature25.numpy()

    feature11 = feature11.squeeze()
    feature12 = feature12.squeeze()
    feature13 = feature13.squeeze()
    feature14 = feature14.squeeze()
    feature15 = feature15.squeeze()

    feature21 = feature21.squeeze()
    feature22 = feature22.squeeze()
    feature23 = feature23.squeeze()
    feature24 = feature24.squeeze()
    feature25 = feature25.squeeze()

    # 第二步，计算余弦相似度

    temp1 = np.linalg.norm(feature11) * np.linalg.norm(feature21)
    temp2 = np.linalg.norm(feature12) * np.linalg.norm(feature22)
    temp3 = np.linalg.norm(feature13) * np.linalg.norm(feature23)

    temp4 = np.linalg.norm(feature14) * np.linalg.norm(feature24)

    temp5 = np.linalg.norm(feature15) * np.linalg.norm(feature25)

    resnet50__specific_layers_Cosine_Similarity = [1, 2, 3, 4, 5]

    resnet50__specific_layers_Cosine_Similarity[0] = np.dot(
        feature11, feature21) / temp1
    resnet50__specific_layers_Cosine_Similarity[1] = np.dot(
        feature12, feature22) / temp2
    resnet50__specific_layers_Cosine_Similarity[2] = np.dot(
        feature13, feature23) / temp3

    resnet50__specific_layers_Cosine_Similarity[3] = np.dot(
        feature14, feature24) / temp4

    resnet50__specific_layers_Cosine_Similarity[4] = np.dot(
        feature15, feature25) / temp5

    return resnet50__specific_layers_Cosine_Similarity

# 计算resnet50__specific_layers网络下两张图片的闵可夫斯基距离5


def compute_resnet50__specific_layers_Minkowski_Distance(
        figpath1, figpath2, num):
    # 第一步，提取特征值
    feature1 = extract_resnet50_specific_layers(figpath1)
    feature11 = feature1[0]
    feature12 = feature1[1]
    feature13 = feature1[2]
    feature14 = feature1[3]
    feature15 = feature1[4]

    feature2 = extract_resnet50_specific_layers(figpath2)
    feature21 = feature2[0]
    feature22 = feature2[1]
    feature23 = feature2[2]
    feature24 = feature2[3]
    feature25 = feature2[4]

    feature11 = feature11.numpy()
    feature12 = feature12.numpy()
    feature13 = feature13.numpy()
    feature14 = feature14.numpy()
    feature15 = feature15.numpy()

    feature21 = feature21.numpy()
    feature22 = feature22.numpy()
    feature23 = feature23.numpy()
    feature24 = feature24.numpy()
    feature25 = feature25.numpy()

    # 第二步，计算闵可夫斯基距离
    resnet50__specific_layers_Minkowski_Distance = [1, 2, 3, 4, 5]
    resnet50__specific_layers_Minkowski_Distance[0] = np.power(
        np.sum(np.power(feature11 - feature21, num)), 1 / num)
    resnet50__specific_layers_Minkowski_Distance[1] = np.power(
        np.sum(np.power(feature12 - feature22, num)), 1 / num)
    resnet50__specific_layers_Minkowski_Distance[2] = np.power(
        np.sum(np.power(feature13 - feature23, num)), 1 / num)
    resnet50__specific_layers_Minkowski_Distance[3] = np.power(
        np.sum(np.power(feature14 - feature24, num)), 1 / num)
    resnet50__specific_layers_Minkowski_Distance[4] = np.power(
        np.sum(np.power(feature15 - feature25, num)), 1 / num)

    return resnet50__specific_layers_Minkowski_Distance

# 计算resnet50__specific_layers网络下两张图片的杰卡德距离6


def compute_resnet50__specific_layers_Jaccard_Distance(
        figpath1, figpath2, threshold):
    # 第一步，提取特征值
    feature1 = extract_resnet50_specific_layers(figpath1)
    feature11 = feature1[0]
    feature12 = feature1[1]
    feature13 = feature1[2]
    feature14 = feature1[3]
    feature15 = feature1[4]

    feature2 = extract_resnet50_specific_layers(figpath2)
    feature21 = feature2[0]
    feature22 = feature2[1]
    feature23 = feature2[2]
    feature24 = feature2[3]
    feature25 = feature2[4]
    resnet50__specific_layers_Jaccard_Distance = [0, 1, 2, 3, 4]
    # 第二步，计算杰卡德距离
    feature11_binary = torch.where(
        feature11 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    feature12_binary = torch.where(
        feature12 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    feature13_binary = torch.where(
        feature13 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    feature14_binary = torch.where(
        feature14 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    feature15_binary = torch.where(
        feature15 > threshold,
        torch.tensor(1),
        torch.tensor(0))

    feature21_binary = torch.where(
        feature21 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    feature22_binary = torch.where(
        feature22 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    feature23_binary = torch.where(
        feature23 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    feature24_binary = torch.where(
        feature24 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    feature25_binary = torch.where(
        feature25 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 计算交集
    intersection1 = torch.sum(torch.min(feature11_binary, feature21_binary))
    intersection2 = torch.sum(torch.min(feature12_binary, feature22_binary))
    intersection3 = torch.sum(torch.min(feature13_binary, feature23_binary))
    intersection4 = torch.sum(torch.min(feature14_binary, feature24_binary))
    intersection5 = torch.sum(torch.min(feature15_binary, feature25_binary))
    # 计算并集
    union1 = torch.sum(torch.max(feature11_binary, feature21_binary))
    union2 = torch.sum(torch.max(feature12_binary, feature22_binary))
    union3 = torch.sum(torch.max(feature13_binary, feature23_binary))
    union4 = torch.sum(torch.max(feature14_binary, feature24_binary))
    union5 = torch.sum(torch.max(feature15_binary, feature25_binary))

    resnet50__specific_layers_Jaccard_Distance[0] = 1.0 - \
        intersection1 / union1
    resnet50__specific_layers_Jaccard_Distance[0] = resnet50__specific_layers_Jaccard_Distance[0].numpy(
    )

    resnet50__specific_layers_Jaccard_Distance[1] = 1.0 - \
        intersection2 / union2
    resnet50__specific_layers_Jaccard_Distance[1] = resnet50__specific_layers_Jaccard_Distance[1].numpy(
    )

    resnet50__specific_layers_Jaccard_Distance[2] = 1.0 - \
        intersection3 / union3
    resnet50__specific_layers_Jaccard_Distance[2] = resnet50__specific_layers_Jaccard_Distance[2].numpy(
    )

    resnet50__specific_layers_Jaccard_Distance[3] = 1.0 - \
        intersection4 / union4
    resnet50__specific_layers_Jaccard_Distance[3] = resnet50__specific_layers_Jaccard_Distance[3].numpy(
    )

    resnet50__specific_layers_Jaccard_Distance[4] = 1.0 - \
        intersection5 / union5
    resnet50__specific_layers_Jaccard_Distance[4] = resnet50__specific_layers_Jaccard_Distance[4].numpy(
    )

    return resnet50__specific_layers_Jaccard_Distance


def all_distance_in_resnet18_50_152_inception_v3_googlenet_mobilenet(
        features1, features2, num, threshold):

    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 1欧式距离
    Euclidean_Distance = np.sqrt(np.sum(np.square(features1 - features2)))
    # 2曼哈顿距离
    Manhattan_Distance = np.sum(np.abs(features1 - features2))
    # 3切比雪夫距离
    Chebyshev_Distance = np.abs(features1 - features2).max()
    # 4余弦相似度
    # 删除冗余维度才能做dot运算
    temp1 = features1.squeeze()
    temp2 = features2.squeeze()
    # 第二步，计算余弦相似度
    temp = np.linalg.norm(temp1) * np.linalg.norm(temp2)
    Cosine_Similarity = np.dot(temp1, temp2) / temp
    # 5闵可夫斯基距离
    Minkowski_Distance = np.power(
        np.sum(
            np.power(
                features1 -
                features2,
                num)),
        1 /
        num)
    # 6杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    Jaccard_Distance = 1.0 - intersection / union
    Jaccard_Distance = Jaccard_Distance.numpy()
    Jaccard_Distance = Jaccard_Distance.item()

    distance = [1.0]
    distance[0] = Euclidean_Distance
    distance.append(Manhattan_Distance)
    distance.append(Chebyshev_Distance)
    distance.append(Cosine_Similarity)
    distance.append(Minkowski_Distance)
    distance.append(Jaccard_Distance)
    return distance


def all_distance_in_alexnet_densenet_vggnet(
        features1, features2, num, threshold):

    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features2_binary = torch.where(
        features2 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    features1_squeezed = features1.squeeze(0)
    features2_squeezed = features2.squeeze(0)
    features1 = features1.numpy()
    features2 = features2.numpy()
    # 1欧式距离
    Euclidean_Distance = np.sqrt(np.sum(np.square(features1 - features2)))
    # 2曼哈顿距离
    Manhattan_Distance = np.sum(np.abs(features1 - features2))
    # 3切比雪夫距离
    Chebyshev_Distance = np.abs(features1 - features2).max()
    # 4余弦相似度
    Cosine_Similarity = F.cosine_similarity(
        features1_squeezed, features2_squeezed, dim=0)
    Cosine_Similarity = Cosine_Similarity.numpy()
    Cosine_Similarity = compute_AVGarray(Cosine_Similarity)
    # 5闵可夫斯基距离
    Minkowski_Distance = np.power(
        np.sum(
            np.power(
                features1 -
                features2,
                num)),
        1 /
        num)
    # 6杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, features2_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, features2_binary))
    Jaccard_Distance = 1.0 - intersection / union
    Jaccard_Distance = Jaccard_Distance.numpy()
    Jaccard_Distance = Jaccard_Distance.item()

    # 将所有距离组成一个数组
    distance = [1.0]
    distance[0] = Euclidean_Distance
    distance.append(Manhattan_Distance)
    distance.append(Chebyshev_Distance)
    distance.append(Cosine_Similarity)
    distance.append(Minkowski_Distance)
    distance.append(Jaccard_Distance)
    return distance

# 注意，5，6，9网络的余弦相似度相关计算是求了张量各个位置余弦相似度的平均值
# 杰卡德距离的最后一层检验效果不好，但是前四层效果还可以


# 输入为图像变量而不是路径
# 1计算resnet18网络下某张图片和一向量的欧式距离1_
def compute_resnet18_with1_Euclidean_Distance_content(features1):

    # 第一步，提取特征值

    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    resnet18_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return resnet18_with1_Euclidean_Distance

# 2计算resnet18网络下某张图片和一向量的余弦相似度1


def compute_resnet18_with1_Cosine_Similarity_content(features1):

    # 提取图片特征值

    # 将张量对应位置的元素全部换为1
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 变为array类型
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 删除冗余维度
    features1 = features1.squeeze()
    replace_one = replace_one.squeeze()

    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(replace_one)
    resnet18_with1_Cosine_Similarity = np.dot(features1, replace_one) / temp

    return resnet18_with1_Cosine_Similarity

# 3计算alexnet网络下某张图片和一向量的欧式距离5


def compute_alexnet_with1_Euclidean_Distance_content(features1):

    # 第一步，提取特征值

    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    alexnet_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return alexnet_with1_Euclidean_Distance

# 4计算resnet50网络下某张图片和一向量的切比雪夫距离2


def compute_resnet50_with1_Chebyshev_Distance_content(features1):
    # 第一步，提取特征值

    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    resnet50_with1_Chebyshev_Distance = np.abs(features1 - replace_one).max()
    return resnet50_with1_Chebyshev_Distance

# 5计算resnet152网络下某张图片和一向量的杰卡德距离3


def compute_resnet152_with1_Jaccard_Distance_content(features1, threshold):
    # 第一步，提取特征值，并将特征值二值化为0和1
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1_binary = torch.where(
        features1 > threshold,
        torch.tensor(1),
        torch.tensor(0))
    replace_one_binary = torch.where(
        replace_one > threshold,
        torch.tensor(1),
        torch.tensor(0))
    # 第二步，计算杰卡德距离
    # 计算交集
    intersection = torch.sum(torch.min(features1_binary, replace_one_binary))
    # 计算并集
    union = torch.sum(torch.max(features1_binary, replace_one_binary))
    resnet152_with1_Jaccard_Distance = 1.0 - intersection / union
    resnet152_with1_Jaccard_Distance = resnet152_with1_Jaccard_Distance.numpy()
    return resnet152_with1_Jaccard_Distance

# 6计算inception_v3网络下某张图片和一向量的曼哈顿距离4


def compute_inception_v3_with1_Manhattan_Distance_content(features1):
    # 第一步，提取特征值
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    inception_v3_with1_Manhattan_Distance = np.sum(
        np.abs(features1 - replace_one))
    return inception_v3_with1_Manhattan_Distance

# 7计算densenet网络下某张图片和一向量的曼哈顿距离6


def compute_densenet_with1_Manhattan_Distance_content(features1):
    # 第一步，提取特征值
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算曼哈顿距离
    densenet_with1_Manhattan_Distance = np.sum(np.abs(features1 - replace_one))
    return densenet_with1_Manhattan_Distance

# 8计算googlenet网络下某张图片和一向量的余弦相似度7


def compute_googlenet_with1_Cosine_Similarity_content(features1):

    # 提取图片特征值

    # 将张量对应位置的元素全部换为1
    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    # 变为array类型
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 删除冗余维度
    features1 = features1.squeeze()
    replace_one = replace_one.squeeze()

    # 第二步，计算余弦相似度
    temp = np.linalg.norm(features1) * np.linalg.norm(replace_one)
    googlenet_with1_Cosine_Similarity = np.dot(features1, replace_one) / temp

    return googlenet_with1_Cosine_Similarity

# 9计算mobilenet网络下某张图片和一向量的切比雪夫距离8


def compute_mobilenet_with1_Chebyshev_Distance_content(features1):
    # 第一步，提取特征值

    replace_one = replace_with_ones_and_keep_as_tensor(features1)
    features1 = features1.numpy()
    replace_one = replace_one.numpy()
   # 第二步，计算切比雪夫距离
    mobilenet_with1_Chebyshev_Distance = np.abs(features1 - replace_one).max()
    return mobilenet_with1_Chebyshev_Distance

# 10计算vggnet网络下某张图片和一向量的欧式距离9


def compute_vggnet_with1_Euclidean_Distance_content(features1):

    # 第一步，提取特征值
    replace_one = replace_with_ones_and_keep_as_tensor(features1)

    features1 = features1.numpy()
    replace_one = replace_one.numpy()
    # 第二步，计算欧式距离
    vggnet_with1_Euclidean_Distance = np.sqrt(
        np.sum(np.square(features1 - replace_one)))
    return vggnet_with1_Euclidean_Distance
