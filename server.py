import io
import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request
from net import Net
import string

app = Flask(__name__)

NUM_CHARS = 4
CHARS = string.ascii_letters + string.digits
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(62)  # 62个类别
    model.to(device)
    model.load_state_dict(torch.load('weight.pth',
                                     map_location=lambda storage, location: storage))
    model.eval()
    print("模型加载成功,可以开始打码")
except Exception as e:
    print(e, "无法使用打码,请确认weight.pth文件是否存在")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes))
    gray_image = image.convert('L')
    img = np.array(gray_image, dtype=np.float32)
    h, w = img.shape
    pred_text = ''
    for i in range(4):
        char_img = img[:, i * (w // NUM_CHARS): (i + 1) * (w // NUM_CHARS)]
        char_img = np.array(Image.fromarray(char_img).resize((32, 32), Image.BICUBIC))
        char_img /= 255.0
        char_img = (char_img - 0.5) / 0.5

        char_img = torch.from_numpy(char_img)

        char_img = char_img.unsqueeze(0)
        char_img = char_img.unsqueeze(0)
        output = model(char_img)[0]
        value, index = torch.max(output, 0)
        pred_text += CHARS[index]

    print(pred_text)
    ret_result = {'pred_text': pred_text}
    return jsonify(ret_result)


if __name__ == '__main__':
    app.run()