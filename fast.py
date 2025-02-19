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

from io import BytesIO
import io
import cairosvg
import pyheif
from .distance import *
from .controller import *

def extract_all_features_with_all_nets_content(images_list):
    """ 提取图像的所有网络特征 """

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')

    if not isinstance(images_list, list):
        images_list = [images_list]
    # 定义每个模型的图像预处理方法
    preprocess_resnet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocess_inception = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocess_alexnet = preprocess_resnet
    preprocess_densenet = preprocess_resnet
    preprocess_googlenet = preprocess_resnet
    preprocess_mobilenet = preprocess_resnet
    preprocess_vggnet = preprocess_resnet

    # 加载所有模型
    model_resnet18 = torch.nn.Sequential(
        *(list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1])).eval()
    model_resnet50 = torch.nn.Sequential(
        *(list(resnet50(weights=ResNet50_Weights.DEFAULT).children())[:-1])).eval()
    model_resnet152 = torch.nn.Sequential(
        *(list(resnet152(weights=ResNet152_Weights.DEFAULT).children())[:-1])).eval()

    model_inception_v3 = inception_v3(
        weights=Inception_V3_Weights.DEFAULT,
        transform_input=False).eval()
    model_inception_v3.aux_logits = False
    model_inception_v3.fc = torch.nn.Identity()

    model_alexnet = torch.nn.Sequential(
        *(list(alexnet(weights=AlexNet_Weights.DEFAULT).children())[:-1])).eval()
    model_densenet121 = torch.nn.Sequential(
        *(list(densenet121(weights=DenseNet121_Weights.DEFAULT).children())[:-1])).eval()

    model_googlenet = googlenet(weights=GoogLeNet_Weights.DEFAULT).eval()
    model_googlenet.aux_logits = False
    model_googlenet.fc = torch.nn.Identity()

    model_mobilenet_v2 = mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT).eval()
    model_mobilenet_v2.classifier[1] = torch.nn.Identity()

    model_vgg19 = torch.nn.Sequential(
        *(list(vgg19(weights=VGG19_Weights.DEFAULT).children())[:-1])).eval()

    # 初始化结果列表
    resnet18_features, resnet50_features, resnet152_features = [], [], []
    inception_v3_features, alexnet_features, densenet_features = [], [], []
    googlenet_features, mobilenet_features, vggnet_features = [], [], []

    total_images = len(images_list)

    try:
        for i, image in enumerate(images_list):
            if not isinstance(image, Image.Image):
                raise ValueError("输入必须是PIL图像对象")

            # 分别为每个模型使用不同的预处理方法
            img_resnet = preprocess_resnet(image)
            img_inception = preprocess_inception(image)
            img_alexnet = preprocess_alexnet(image)
            img_densenet = preprocess_densenet(image)
            img_googlenet = preprocess_googlenet(image)
            img_mobilenet = preprocess_mobilenet(image)
            img_vggnet = preprocess_vggnet(image)

            batch_resnet = torch.unsqueeze(img_resnet, 0)
            batch_inception = torch.unsqueeze(img_inception, 0)
            batch_alexnet = torch.unsqueeze(img_alexnet, 0)
            batch_densenet = torch.unsqueeze(img_densenet, 0)
            batch_googlenet = torch.unsqueeze(img_googlenet, 0)
            batch_mobilenet = torch.unsqueeze(img_mobilenet, 0)
            batch_vggnet = torch.unsqueeze(img_vggnet, 0)

            with torch.no_grad():
                # 提取特征
                resnet18_features.append(model_resnet18(batch_resnet))
                resnet50_features.append(model_resnet50(batch_resnet))
                resnet152_features.append(model_resnet152(batch_resnet))
                inception_v3_features.append(
                    model_inception_v3(batch_inception))
                alexnet_features.append(model_alexnet(batch_alexnet))
                densenet_features.append(model_densenet121(batch_densenet))
                googlenet_features.append(model_googlenet(batch_googlenet))
                mobilenet_features.append(model_mobilenet_v2(batch_mobilenet))
                vggnet_features.append(model_vgg19(batch_vggnet))

            # 打印当前处理进度
            logging.info(f'Processed image {i + 1}/{total_images}')

        return resnet18_features, resnet50_features, resnet152_features, \
            inception_v3_features, alexnet_features, densenet_features, \
            googlenet_features, mobilenet_features, vggnet_features

    except ValueError as e:
        logging.error("Error: %s", e)
        return None, None, None, None, None, None, None, None, None
    except Exception as e:
        logging.error("An error occurred: %s", e)
        return None, None, None, None, None, None, None, None, None

def open_heic_image(image_path):
    # 使用 pyheif 读取 HEIC 文件并转换为 PIL 图像
    heif_file = pyheif.read(image_path)
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    return image


def extract_all_features_with_all_nets(figpath):

    features1 = extract_resnet18(figpath)
    features2 = extract_resnet50(figpath)
    features3 = extract_resnet152(figpath)
    features4 = extract_inception_v3(figpath)
    features5 = extract_alexnet(figpath)
    features6 = extract_densenet(figpath)
    features7 = extract_googlenet(figpath)
    features8 = extract_mobilenet(figpath)
    features9 = extract_vggnet(figpath)
    return features1, features2, features3, features4, features5, features6, features7, features8, features9

def extract_picture_all_features(filepath):

    # 如果为heic图像
    if filepath.lower().endswith('.heic'):
        pil_image = open_heic_image(filepath)

        # 多网络的特征
        resnet18, resnet50, resnet152, inception_v3, alexnet, \
            densenet, googlenet, mobilenet, vggnet = extract_all_features_with_all_nets_content(pil_image)
        # 与1向量的距离
        # 和1向量的距离,共十个
        # 杰卡德距离的参数为0
        threshold = 0
        # 计算distance_with1_resnet18_Cosine等一共十个值
        distance_with1_resnet18_Euclidean = compute_resnet18_with1_Euclidean_Distance_content(
            pil_image)
        distance_with1_resnet18_Cosine = compute_resnet18_with1_Cosine_Similarity_content(
            pil_image)
        distance_with1_alexnet_Euclidean = compute_alexnet_with1_Euclidean_Distance_content(
            pil_image)
        distance_with1_resnet50_Chebyshev = compute_resnet50_with1_Chebyshev_Distance_content(
            pil_image)
        distance_with1_resnet152_Jaccard = compute_resnet152_with1_Jaccard_Distance_content(
            pil_image, threshold)
        distance_with1_inception_v3_Manhattan = compute_inception_v3_with1_Manhattan_Distance_content(
            pil_image)
        distance_with1_densenet_Manhattan = compute_densenet_with1_Manhattan_Distance_content(
            pil_image)
        distance_with1_googlenet_Cosine = compute_googlenet_with1_Cosine_Similarity_content(
            pil_image)
        distance_with1_mobilenet_Chebyshev = compute_mobilenet_with1_Chebyshev_Distance_content(
            pil_image)
        distance_with1_vggnet_Euclidean = compute_vggnet_with1_Euclidean_Distance_content(
            pil_image)
        return resnet18, resnet50, resnet152, inception_v3, alexnet, densenet, \
            googlenet, mobilenet, vggnet, distance_with1_resnet18_Euclidean, distance_with1_resnet18_Cosine, \
            distance_with1_alexnet_Euclidean, distance_with1_resnet50_Chebyshev, distance_with1_resnet152_Jaccard, \
            distance_with1_inception_v3_Manhattan, distance_with1_densenet_Manhattan, distance_with1_googlenet_Cosine, \
            distance_with1_mobilenet_Chebyshev, distance_with1_vggnet_Euclidean
    # 如果为svg图像
    elif filepath.lower().endswith('.svg'):
        # 读取SVG内容
        with open(filepath, 'r', encoding='utf-8') as file:
            svg_content = file.read()
        # 转换为PNG并保存到内存中
        png_output = BytesIO()
        cairosvg.svg2png(bytestring=svg_content, write_to=png_output)
        # 将内存中的PNG数据加载为PIL Image对象
        png_output.seek(0)  # 复位流位置到开始
        pil_image = Image.open(png_output)

        # 多网络的特征
        resnet18, resnet50, resnet152, inception_v3, alexnet, \
            densenet, googlenet, mobilenet, vggnet = extract_all_features_with_all_nets_content(pil_image)
        # 与1向量的距离
        # 和1向量的距离,共十个
        # 杰卡德距离的参数为0
        threshold = 0
        # 计算distance_with1_resnet18_Cosine等一共十个值
        distance_with1_resnet18_Euclidean = compute_resnet18_with1_Euclidean_Distance_content(
            pil_image)
        distance_with1_resnet18_Cosine = compute_resnet18_with1_Cosine_Similarity_content(
            pil_image)
        distance_with1_alexnet_Euclidean = compute_alexnet_with1_Euclidean_Distance_content(
            pil_image)
        distance_with1_resnet50_Chebyshev = compute_resnet50_with1_Chebyshev_Distance_content(
            pil_image)
        distance_with1_resnet152_Jaccard = compute_resnet152_with1_Jaccard_Distance_content(
            pil_image, threshold)
        distance_with1_inception_v3_Manhattan = compute_inception_v3_with1_Manhattan_Distance_content(
            pil_image)
        distance_with1_densenet_Manhattan = compute_densenet_with1_Manhattan_Distance_content(
            pil_image)
        distance_with1_googlenet_Cosine = compute_googlenet_with1_Cosine_Similarity_content(
            pil_image)
        distance_with1_mobilenet_Chebyshev = compute_mobilenet_with1_Chebyshev_Distance_content(
            pil_image)
        distance_with1_vggnet_Euclidean = compute_vggnet_with1_Euclidean_Distance_content(
            pil_image)
        return resnet18, resnet50, resnet152, inception_v3, alexnet, densenet, \
            googlenet, mobilenet, vggnet, distance_with1_resnet18_Euclidean, distance_with1_resnet18_Cosine, \
            distance_with1_alexnet_Euclidean, distance_with1_resnet50_Chebyshev, distance_with1_resnet152_Jaccard, \
            distance_with1_inception_v3_Manhattan, distance_with1_densenet_Manhattan, distance_with1_googlenet_Cosine, \
            distance_with1_mobilenet_Chebyshev, distance_with1_vggnet_Euclidean
    # 如果为eps图像
    elif filepath.lower().endswith('.eps'):
        # 打开EPS文件
        with Image.open(filepath) as img:
            # 将图像转换为RGB模式（默认是CMYK或其他模式）
            img = img.convert("RGB")

            # 保存为PNG格式到内存中
            png_output = io.BytesIO()
            img.save(png_output, format='PNG')

            # 将内存中的PNG数据加载为PIL Image对象
            png_output.seek(0)  # 复位流位置到开始
            pil_image = Image.open(png_output)

            # 多网络的特征
            resnet18, resnet50, resnet152, inception_v3, alexnet, \
                densenet, googlenet, mobilenet, vggnet = extract_all_features_with_all_nets_content(pil_image)
            # 与1向量的距离
            # 和1向量的距离,共十个
            # 杰卡德距离的参数为0
            threshold = 0
            # 计算distance_with1_resnet18_Cosine等一共十个值
            distance_with1_resnet18_Euclidean = compute_resnet18_with1_Euclidean_Distance_content(
                pil_image)
            distance_with1_resnet18_Cosine = compute_resnet18_with1_Cosine_Similarity_content(
                pil_image)
            distance_with1_alexnet_Euclidean = compute_alexnet_with1_Euclidean_Distance_content(
                pil_image)
            distance_with1_resnet50_Chebyshev = compute_resnet50_with1_Chebyshev_Distance_content(
                pil_image)
            distance_with1_resnet152_Jaccard = compute_resnet152_with1_Jaccard_Distance_content(
                pil_image, threshold)
            distance_with1_inception_v3_Manhattan = compute_inception_v3_with1_Manhattan_Distance_content(
                pil_image)
            distance_with1_densenet_Manhattan = compute_densenet_with1_Manhattan_Distance_content(
                pil_image)
            distance_with1_googlenet_Cosine = compute_googlenet_with1_Cosine_Similarity_content(
                pil_image)
            distance_with1_mobilenet_Chebyshev = compute_mobilenet_with1_Chebyshev_Distance_content(
                pil_image)
            distance_with1_vggnet_Euclidean = compute_vggnet_with1_Euclidean_Distance_content(
                pil_image)
            return resnet18, resnet50, resnet152, inception_v3, alexnet, densenet, \
                googlenet, mobilenet, vggnet, distance_with1_resnet18_Euclidean, distance_with1_resnet18_Cosine, \
                distance_with1_alexnet_Euclidean, distance_with1_resnet50_Chebyshev, distance_with1_resnet152_Jaccard, \
                distance_with1_inception_v3_Manhattan, distance_with1_densenet_Manhattan, distance_with1_googlenet_Cosine, \
                distance_with1_mobilenet_Chebyshev, distance_with1_vggnet_Euclidean
    # 如果为ico图像
    elif filepath.lower().endswith('.ico'):
        # 打开ICO文件
        with Image.open(filepath) as img:
            # 如果ICO文件包含多个分辨率，选择其中一个
            img = img.convert("RGBA")  # 转换为RGBA模式以保留透明度

            # 保存为PNG格式到内存中
            png_output = io.BytesIO()
            img.save(png_output, format='PNG')

            # 将内存中的PNG数据加载为PIL Image对象
            png_output.seek(0)  # 复位流位置到开始
            pil_image = Image.open(png_output)
            
            # 多网络的特征
            resnet18, resnet50, resnet152, inception_v3, alexnet, \
                densenet, googlenet, mobilenet, vggnet = extract_all_features_with_all_nets_content(pil_image)
            # 与1向量的距离
            # 和1向量的距离,共十个
            # 杰卡德距离的参数为0
            threshold = 0
            # 计算distance_with1_resnet18_Cosine等一共十个值
            distance_with1_resnet18_Euclidean = compute_resnet18_with1_Euclidean_Distance_content(
                pil_image)
            distance_with1_resnet18_Cosine = compute_resnet18_with1_Cosine_Similarity_content(
                pil_image)
            distance_with1_alexnet_Euclidean = compute_alexnet_with1_Euclidean_Distance_content(
                pil_image)
            distance_with1_resnet50_Chebyshev = compute_resnet50_with1_Chebyshev_Distance_content(
                pil_image)
            distance_with1_resnet152_Jaccard = compute_resnet152_with1_Jaccard_Distance_content(
                pil_image, threshold)
            distance_with1_inception_v3_Manhattan = compute_inception_v3_with1_Manhattan_Distance_content(
                pil_image)
            distance_with1_densenet_Manhattan = compute_densenet_with1_Manhattan_Distance_content(
                pil_image)
            distance_with1_googlenet_Cosine = compute_googlenet_with1_Cosine_Similarity_content(
                pil_image)
            distance_with1_mobilenet_Chebyshev = compute_mobilenet_with1_Chebyshev_Distance_content(
                pil_image)
            distance_with1_vggnet_Euclidean = compute_vggnet_with1_Euclidean_Distance_content(
                pil_image)
            return resnet18, resnet50, resnet152, inception_v3, alexnet, densenet, \
                googlenet, mobilenet, vggnet, distance_with1_resnet18_Euclidean, distance_with1_resnet18_Cosine, \
                distance_with1_alexnet_Euclidean, distance_with1_resnet50_Chebyshev, distance_with1_resnet152_Jaccard, \
                distance_with1_inception_v3_Manhattan, distance_with1_densenet_Manhattan, distance_with1_googlenet_Cosine, \
                distance_with1_mobilenet_Chebyshev, distance_with1_vggnet_Euclidean
    # 如果为其他图像
    else:
        num = 4
        threshold = 0

        
        # 提取高维特征向量
        features11, features12, features13, features14, features15, features16, \
            features17, features18, features19 = extract_all_features_with_all_nets(filepath)

        # 计算distance_with1_resnet18_Cosine等一共十个值
        distance_with1_resnet18_Euclidean = compute_resnet18_with1_Euclidean_Distance(
            filepath)
        distance_with1_resnet18_Cosine = compute_resnet18_with1_Cosine_Similarity(
            filepath)
        distance_with1_alexnet_Euclidean = compute_alexnet_with1_Euclidean_Distance(
            filepath)
        distance_with1_resnet50_Chebyshev = compute_resnet50_with1_Chebyshev_Distance(
            filepath)
        distance_with1_resnet152_Jaccard = compute_resnet152_with1_Jaccard_Distance(
            filepath, threshold)
        distance_with1_inception_v3_Manhattan = compute_inception_v3_with1_Manhattan_Distance(
            filepath)
        distance_with1_densenet_Manhattan = compute_densenet_with1_Manhattan_Distance(
            filepath)
        distance_with1_googlenet_Cosine = compute_googlenet_with1_Cosine_Similarity(
            filepath)
        distance_with1_mobilenet_Chebyshev = compute_mobilenet_with1_Chebyshev_Distance(
            filepath)
        distance_with1_vggnet_Euclidean = compute_vggnet_with1_Euclidean_Distance(
            filepath)

        return features11, features12, features13, features14, features15, features16, \
            features17, features18, features19, distance_with1_resnet18_Euclidean, distance_with1_resnet18_Cosine, \
            distance_with1_alexnet_Euclidean, distance_with1_resnet50_Chebyshev, distance_with1_resnet152_Jaccard, \
            distance_with1_inception_v3_Manhattan, distance_with1_densenet_Manhattan, distance_with1_googlenet_Cosine, \
            distance_with1_mobilenet_Chebyshev, distance_with1_vggnet_Euclidean
    


def Picture_find_suspiciousID(
        distance_with1_resnet18_Euclidean,
        distance_with1_resnet18_Cosine,
        distance_with1_alexnet_Euclidean,
        distance_with1_resnet50_Chebyshev,
        distance_with1_resnet152_Jaccard,
        distance_with1_inception_v3_Manhattan,
        distance_with1_densenet_Manhattan,
        distance_with1_googlenet_Cosine,
        distance_with1_mobilenet_Chebyshev,
        distance_with1_vggnet_Euclidean):
    c = Controller()
    DataID1 = c.FindDataID_and_resnet18_with1_Cosine_Similarity(
        distance_with1_resnet18_Cosine)
    DataID1 = [(x[0], abs(x[1] - distance_with1_resnet18_Cosine))
               for x in DataID1]
    DataID1_sorted = sorted(DataID1, key=lambda x: x[1])

    DataID2 = c.FindDataID_and_alexnet_with1_Euclidean_Distance(
        distance_with1_alexnet_Euclidean)
    DataID2 = [(x[0], abs(x[1] - distance_with1_alexnet_Euclidean))
               for x in DataID2]
    DataID2_sorted = sorted(DataID2, key=lambda x: x[1])

    DataID3 = c.FindDataID_and_resnet18_with1_Euclidean_Distance(
        distance_with1_resnet18_Euclidean)
    DataID3 = [(x[0], abs(x[1] - distance_with1_resnet18_Euclidean))
               for x in DataID3]
    DataID3_sorted = sorted(DataID3, key=lambda x: x[1])

    DataID4 = c.FindDataID_and_resnet50_with1_Chebyshev_Distance(
        distance_with1_resnet50_Chebyshev)
    DataID4 = [(x[0], abs(x[1] - distance_with1_resnet50_Chebyshev))
               for x in DataID4]
    DataID4_sorted = sorted(DataID4, key=lambda x: x[1])

    DataID5 = c.FindDataID_and_resnet152_with1_Jaccard_Distance(
        distance_with1_resnet152_Jaccard)
    DataID5 = [(x[0], abs(x[1] - distance_with1_resnet152_Jaccard))
               for x in DataID5]
    DataID5_sorted = sorted(DataID5, key=lambda x: x[1])

    DataID6 = c.FindDataID_and_inception_v3_with1_Manhattan_Distance(
        distance_with1_inception_v3_Manhattan)
    DataID6 = [(x[0], abs(x[1] - distance_with1_inception_v3_Manhattan))
               for x in DataID6]
    DataID6_sorted = sorted(DataID6, key=lambda x: x[1])

    DataID7 = c.FindDataID_and_densenet_with1_Manhattan_Distance(
        distance_with1_densenet_Manhattan)
    DataID7 = [(x[0], abs(x[1] - distance_with1_densenet_Manhattan))
               for x in DataID7]
    DataID7_sorted = sorted(DataID7, key=lambda x: x[1])

    DataID8 = c.FindDataID_and_googlenet_with1_Cosine_Similarity(
        distance_with1_googlenet_Cosine)
    DataID8 = [(x[0], abs(x[1] - distance_with1_googlenet_Cosine))
               for x in DataID8]
    DataID8_sorted = sorted(DataID8, key=lambda x: x[1])

    DataID9 = c.FindDataID_and_mobilenet_with1_Chebyshev_Distance(
        distance_with1_mobilenet_Chebyshev)
    DataID9 = [(x[0], abs(x[1] - distance_with1_mobilenet_Chebyshev))
               for x in DataID9]
    DataID9_sorted = sorted(DataID9, key=lambda x: x[1])

    DataID10 = c.FindDataID_and_vggnet_with1_Euclidean_Distance(
        distance_with1_vggnet_Euclidean)
    DataID10 = [(x[0], abs(x[1] - distance_with1_vggnet_Euclidean))
                for x in DataID10]
    DataID10_sorted = sorted(DataID10, key=lambda x: x[1])

    q = 20
    dataID_first_201 = [item[0] for item in DataID1_sorted[:q]]
    dataID_first_202 = [item[0] for item in DataID2_sorted[:q]]
    dataID_first_203 = [item[0] for item in DataID3_sorted[:q]]
    dataID_first_204 = [item[0] for item in DataID4_sorted[:q]]
    dataID_first_205 = [item[0] for item in DataID5_sorted[:q]]
    dataID_first_206 = [item[0] for item in DataID6_sorted[:q]]
    dataID_first_207 = [item[0] for item in DataID7_sorted[:q]]
    dataID_first_208 = [item[0] for item in DataID8_sorted[:q]]
    dataID_first_209 = [item[0] for item in DataID9_sorted[:q]]
    dataID_first_2010 = [item[0] for item in DataID10_sorted[:q]]

    dataID = set(dataID_first_201).union(dataID_first_202).union(dataID_first_203).union(dataID_first_204).union(dataID_first_205)\
        .union(dataID_first_206).union(dataID_first_207).union(dataID_first_208).union(dataID_first_209).union(dataID_first_2010)

    # print(dataID_first_201)
    # print(dataID_first_202)
    # print(dataID_first_203)
    # print(dataID_first_204)
    # print(dataID_first_205)
    # print(dataID_first_206)
    # print(dataID_first_207)
    # print(dataID_first_208)
    # print(dataID_first_209)
    # print(dataID_first_2010)

    """
    alldataID = dataID_first_201
    alldataID.extend(dataID_first_202)
    alldataID.extend(dataID_first_203)
    alldataID.extend(dataID_first_204)
    alldataID.extend(dataID_first_205)
    alldataID.extend(dataID_first_206)
    alldataID.extend(dataID_first_207)
    alldataID.extend(dataID_first_208)
    alldataID.extend(dataID_first_209)
    alldataID.extend(dataID_first_2010)
    print(alldataID)
    k = 2  # 定义超过k次的阈值
    counter = Counter(alldataID)
    result = list(set([num for num in alldataID if counter[num] > k]))
    print(result)
    print(len(result))
    """

    DataID = list(dataID)
    print(DataID)
    # print(len(DataID))

    return (DataID)