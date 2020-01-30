import io
import requests
import numpy as np
import cv2
import json
from PIL import Image
import random


server_url = "http://127.0.0.1:5000/predict"
while True:
    img_url = f"https://wap.gd.10086.cn/nwap/card/cardOrder/imageCode.jsps?id={random.randint(0, 100)}"

    img_response = requests.get(img_url, timeout=(3, 7))

    code_img = img_response.content

    res = requests.post(server_url, files={'file': code_img})
    pred_text = json.loads(res.text)['pred_text']

    # 这部分代码用于debug
    image = Image.open(io.BytesIO(code_img))
    gray_image = image.convert('L')
    img = np.array(gray_image, dtype=np.uint8)

    print(pred_text)
    cv2.imshow(' ', img)
    cv2.waitKey(0)