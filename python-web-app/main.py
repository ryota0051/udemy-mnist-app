from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import onnxruntime


app = FastAPI()

app.mount('/static', StaticFiles(directory='static'))


@app.get('/', response_class=HTMLResponse)
async def root():
    with open('index.html', 'r') as f:
        return f.read()


@app.post('/api/predict')
async def predict(image: UploadFile = File(...)):
    image_data = await image.read()
    pil_image = Image.open(BytesIO(image_data))
    # preprocess
    resized_image = pil_image.resize((28, 28))
    resized_arr = np.array(resized_image)
    transposed_arr = resized_arr.transpose(2, 0, 1)
    alpha_arr: np.ndarray = transposed_arr[3]
    reshaped_arr = alpha_arr.reshape(-1)
    input_ = [reshaped_arr.astype(np.float32)]
    # predict
    onnx_session = onnxruntime.InferenceSession('model.onnx')
    output = onnx_session.run(['probabilities'], {'float_input': input_})
    result: np.ndarray = output[0][0]

    return {'probabilities': result.tolist()}
