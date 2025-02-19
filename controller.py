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


import pymysql


class Controller(object):
    def __init__(self):
        self.conn = pymysql.connect(
            host='localhost',
            port='xxx',
            user='root',
            passwd='xxx',
            charset='utf8',
            db='xxx'
        )
        self.cur = self.conn.cursor()

    def __del__(self):
        self.cur.close()
        self.conn.close()


# 1图片类型数据，依据resnet18_with1_Cosine_Similarity查找可能侵权的数据ID
    def FindDataID_and_resnet18_with1_Cosine_Similarity(
            self, distance_with1_resnet18_cosine):
        # 从 Config 表中根据 ConfigID=1 查询 picture_fastselect_1 的值
        config_sql = "SELECT picture_fastselect_1 FROM Config WHERE ConfigID=1"
        try:
            self.cur.execute(config_sql)
            picture_fastselect_1 = self.cur.fetchone()[0]  # 获取查询结果的第一个值
        except BaseException:
            return False

        # 使用查询到的 picture_fastselect_1 值代替硬编码的 0.023
        up = distance_with1_resnet18_cosine + picture_fastselect_1
        down = distance_with1_resnet18_cosine - picture_fastselect_1
        # 查询 Picture_HighVector 表
        sql = "SELECT DataID, distance_with1_resnet18_Cosine FROM Picture_HighVector WHERE distance_with1_resnet18_cosine BETWEEN %s and %s"
        try:
            self.cur.execute(sql, (down, up))
            res = self.cur.fetchall()
            return res
        except BaseException:
            return False

    # 2图片类型数据，依据alexnet_with1_Euclidean_Distance查找可能侵权的数据ID
    def FindDataID_and_alexnet_with1_Euclidean_Distance(
            self, distance_with1_alexnet_Euclidean):
        # 从 Config 表中根据 ConfigID=1 查询 picture_fastselect_2 的值
        config_sql = "SELECT picture_fastselect_2 FROM Config WHERE ConfigID=1"
        try:
            self.cur.execute(config_sql)
            picture_fastselect_2 = self.cur.fetchone()[0]  # 获取查询结果的第一个值
        except BaseException:
            return False

        # 使用查询到的 picture_fastselect_2 值代替硬编码的 37
        up = distance_with1_alexnet_Euclidean + picture_fastselect_2
        down = distance_with1_alexnet_Euclidean - picture_fastselect_2

        # 查询 Picture_HighVector 表
        sql = "SELECT DataID, distance_with1_alexnet_Euclidean FROM Picture_HighVector WHERE distance_with1_alexnet_Euclidean BETWEEN %s and %s"
        try:
            self.cur.execute(sql, (down, up))
            res = self.cur.fetchall()
            return res
        except BaseException:
            return False

    # 3图片类型数据，依据resxnet_with1_Euclidean_Distance查找可能侵权的数据ID
    def FindDataID_and_resnet18_with1_Euclidean_Distance(
            self, distance_with1_resnet18_Euclidean):
        # 从 Config 表中根据 ConfigID=1 查询 picture_fastselect_3 的值
        config_sql = "SELECT picture_fastselect_3 FROM Config WHERE ConfigID=1"
        try:
            self.cur.execute(config_sql)
            picture_fastselect_3 = self.cur.fetchone()[0]  # 获取查询结果的第一个值
        except BaseException:
            return False

        # 使用查询到的 picture_fastselect_3 值代替硬编码的 1.97
        up = distance_with1_resnet18_Euclidean + picture_fastselect_3
        down = distance_with1_resnet18_Euclidean - picture_fastselect_3

        # 查询 Picture_HighVector 表
        sql = "SELECT DataID, distance_with1_resnet18_Euclidean FROM Picture_HighVector WHERE distance_with1_resnet18_Euclidean BETWEEN %s and %s"
        try:
            self.cur.execute(sql, (down, up))
            res = self.cur.fetchall()
            return res
        except BaseException:
            return False

    # 4图片类型数据，依据distance_with1_resnet50_Chebyshev查找可能侵权的数据ID
    def FindDataID_and_resnet50_with1_Chebyshev_Distance(
            self, distance_with1_resnet50_Chebyshev):
        # 从 Config 表中根据 ConfigID=1 查询 picture_fastselect_4 的值
        config_sql = "SELECT picture_fastselect_4 FROM Config WHERE ConfigID=1"
        try:
            self.cur.execute(config_sql)
            picture_fastselect_4 = self.cur.fetchone()[0]  # 获取查询结果的第一个值
        except BaseException:
            return False

        # 使用查询到的 picture_fastselect_4 值代替硬编码的 1.06
        up = distance_with1_resnet50_Chebyshev + picture_fastselect_4
        down = distance_with1_resnet50_Chebyshev - picture_fastselect_4

        # 查询 Picture_HighVector 表
        sql = "SELECT DataID, distance_with1_resnet50_Chebyshev FROM Picture_HighVector WHERE distance_with1_resnet50_Chebyshev BETWEEN %s and %s"
        try:
            self.cur.execute(sql, (down, up))
            res = self.cur.fetchall()
            return res
        except BaseException:
            return False

    # 5图片类型数据，依据distance_with1_resnet152_Jaccard查找可能侵权的数据ID
    def FindDataID_and_resnet152_with1_Jaccard_Distance(
            self, distance_with1_resnet152_Jaccard):
        # 从 Config 表中根据 ConfigID=1 查询 picture_fastselect_5 的值
        config_sql = "SELECT picture_fastselect_5 FROM Config WHERE ConfigID=1"
        try:
            self.cur.execute(config_sql)
            picture_fastselect_5 = self.cur.fetchone()[0]  # 获取查询结果的第一个值
        except BaseException:
            return False

        # 使用查询到的 picture_fastselect_5 值代替硬编码的 0.007
        up = distance_with1_resnet152_Jaccard + picture_fastselect_5
        down = distance_with1_resnet152_Jaccard - picture_fastselect_5

        # 查询 Picture_HighVector 表
        sql = "SELECT DataID, distance_with1_resnet152_Jaccard FROM Picture_HighVector WHERE distance_with1_resnet152_Jaccard BETWEEN %s and %s"
        try:
            self.cur.execute(sql, (down, up))
            res = self.cur.fetchall()
            return res
        except BaseException:
            return False

    # 6图片类型数据，依据distance_with1_inception_v3_Manhattan查找可能侵权的数据ID
    def FindDataID_and_inception_v3_with1_Manhattan_Distance(
            self, distance_with1_inception_v3_Manhattan):
        # 从 Config 表中根据 ConfigID=1 查询 picture_fastselect_6 的值
        config_sql = "SELECT picture_fastselect_6 FROM Config WHERE ConfigID=1"
        try:
            self.cur.execute(config_sql)
            picture_fastselect_6 = self.cur.fetchone()[0]  # 获取查询结果的第一个值
        except BaseException:
            return False

        # 使用查询到的 picture_fastselect_6 值代替硬编码的 143
        up = distance_with1_inception_v3_Manhattan + picture_fastselect_6
        down = distance_with1_inception_v3_Manhattan - picture_fastselect_6

        # 查询 Picture_HighVector 表
        sql = "SELECT DataID, distance_with1_inception_v3_Manhattan FROM Picture_HighVector WHERE distance_with1_inception_v3_Manhattan BETWEEN %s and %s"
        try:
            self.cur.execute(sql, (down, up))
            res = self.cur.fetchall()
            return res
        except BaseException:
            return False

    # 7图片类型数据，依据distance_with1_densenet_Manhattan查找可能侵权的数据ID
    def FindDataID_and_densenet_with1_Manhattan_Distance(
            self, distance_with1_densenet_Manhattan):
        # 从 Config 表中根据 ConfigID=1 查询 picture_fastselect_7 的值
        config_sql = "SELECT picture_fastselect_7 FROM Config WHERE ConfigID=1"
        try:
            self.cur.execute(config_sql)
            picture_fastselect_7 = self.cur.fetchone()[0]  # 获取查询结果的第一个值
        except BaseException:
            return False

            # 使用查询到的 picture_fastselect_7 值代替硬编码的 2150
        up = distance_with1_densenet_Manhattan + picture_fastselect_7
        down = distance_with1_densenet_Manhattan - picture_fastselect_7

        # 查询 Picture_HighVector 表
        sql = "SELECT DataID, distance_with1_densenet_Manhattan FROM Picture_HighVector WHERE distance_with1_densenet_Manhattan BETWEEN %s and %s"
        try:
            self.cur.execute(sql, (down, up))
            res = self.cur.fetchall()
            return res
        except BaseException:
            return False

    # 8图片类型数据，依据distance_with1_googlenet_Cosine查找可能侵权的数据ID
    def FindDataID_and_googlenet_with1_Cosine_Similarity(
            self, distance_with1_googlenet_Cosine):
        # 从 Config 表中根据 ConfigID=1 查询 picture_fastselect_8 的值
        config_sql = "SELECT picture_fastselect_8 FROM Config WHERE ConfigID=1"
        try:
            self.cur.execute(config_sql)
            picture_fastselect_8 = self.cur.fetchone()[0]  # 获取查询结果的第一个值
        except BaseException:
            return False

        # 使用查询到的 picture_fastselect_8 值代替硬编码的 0.045
        up = distance_with1_googlenet_Cosine + picture_fastselect_8
        down = distance_with1_googlenet_Cosine - picture_fastselect_8

        # 查询 Picture_HighVector 表
        sql = "SELECT DataID, distance_with1_googlenet_Cosine FROM Picture_HighVector WHERE distance_with1_googlenet_Cosine BETWEEN %s and %s"
        try:
            self.cur.execute(sql, (down, up))
            res = self.cur.fetchall()
            return res
        except BaseException:
            return False

    # 9图片类型数据，依据distance_with1_mobilenet_Chebyshev查找可能侵权的数据ID
    def FindDataID_and_mobilenet_with1_Chebyshev_Distance(
            self, distance_with1_mobilenet_Chebyshev):
        # 从 Config 表中根据 ConfigID=1 查询 picture_fastselect_9 的值
        config_sql = "SELECT picture_fastselect_9 FROM Config WHERE ConfigID=1"
        try:
            self.cur.execute(config_sql)
            picture_fastselect_9 = self.cur.fetchone()[0]  # 获取查询结果的第一个值
        except BaseException:
            return False

        # 使用查询到的 picture_fastselect_9 值代替硬编码的 0.58
        up = distance_with1_mobilenet_Chebyshev + picture_fastselect_9
        down = distance_with1_mobilenet_Chebyshev - picture_fastselect_9

        # 查询 Picture_HighVector 表
        sql = "SELECT DataID, distance_with1_mobilenet_Chebyshev FROM Picture_HighVector WHERE distance_with1_mobilenet_Chebyshev BETWEEN %s and %s"
        try:
            self.cur.execute(sql, (down, up))
            res = self.cur.fetchall()
            return res
        except BaseException:
            return False

    # 10图片类型数据，依据distance_with1_vggnet_Euclidean查找可能侵权的数据ID
    def FindDataID_and_vggnet_with1_Euclidean_Distance(
            self, distance_with1_vggnet_Euclidean):
        # 从 Config 表中根据 ConfigID=1 查询 picture_fastselect_10 的值
        config_sql = "SELECT picture_fastselect_10 FROM Config WHERE ConfigID=1"
        try:
            self.cur.execute(config_sql)
            picture_fastselect_10 = self.cur.fetchone()[0]  # 获取查询结果的第一个值
        except BaseException:
            return False

        # 使用查询到的 picture_fastselect_10 值代替硬编码的 14.2
        up = distance_with1_vggnet_Euclidean + picture_fastselect_10
        down = distance_with1_vggnet_Euclidean - picture_fastselect_10

        # 查询 Picture_HighVector 表
        sql = "SELECT DataID, distance_with1_vggnet_Euclidean FROM Picture_HighVector WHERE distance_with1_vggnet_Euclidean BETWEEN %s and %s"
        try:
            self.cur.execute(sql, (down, up))
            res = self.cur.fetchall()
            return res
        except BaseException:
            return False