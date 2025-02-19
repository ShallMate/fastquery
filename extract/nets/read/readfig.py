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

import logging
from PIL import Image
import os


def read_image(image_path):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        # Check if the file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError("File does not exist")

        # Open the image
        image = Image.open(image_path)
        return image
    except FileNotFoundError as e:
        logging.error("Error: %s", e)
        return None
    except Exception as e:
        logging.error("An error occurred: %s", e)
        return None


'''
# Image file path
image_path = "/user/lgw/CORProject/datas/figures/1.tif"

# Open the image
read_image(image_path)
'''
