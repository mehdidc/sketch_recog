import sys
import json
import click

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.filters import threshold_otsu
from flask import Flask, request, redirect, render_template, url_for, Response, jsonify
import base64

import numpy as np
import pandas as pd

from machinepredict.interface import load
from machinepredict.interface import predict as model_predict

model_name = 'hwrt'

if model_name == 'hwrt':
    model = load('models/hwrt')
    data = np.load('models/hwrt.npz')
    class_to_name = dict(zip(data['y'], data['y_str']))
elif model_name == 'digits_and_letters':
    model = load('models/digits_and_letters')
    class_to_name = dict(zip(range(36), '0123456789abcdefghijklmnopqrstuvwxyz'))
else:
    raise ValueError(model_name)

DEBUG = True
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

def autocrop(img):
    r, c = np.indices(img.shape)
    r = ((img>0) * r).flatten()
    r = r[r > 0]
    c = ((img>0) * c)
    c = c[c>0]
    ru, rs = r.mean(), r.std()
    cu, cs = c.mean(), c.std()
    r = int(ru - rs)
    c = int(cu - cs)
    s = int(max(rs, cs) * 3)
    return img[r - s:r + s, c - s:c + s]
    



@click.command()
@click.option('--host', default='0.0.0.0', required=False)
@click.option('--port', default=20004, required=False)
def serve(host, port):
    app.run(host=host, port=port, debug=DEBUG)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('sketch.html')

@app.route('/predict')
def predict():
    img = request.args.get('img', '', type=str)
    img = process_img(img)
    X = img[np.newaxis, np.newaxis, :, :]
    y = model_predict(model, X)
    class_id = np.argsort(y, axis=1)[0][::-1]
    names = [class_to_name[cl] for cl in class_id]
    probas = [float(y[0, cl]) for cl in class_id]
    out = {'names': names, 'probas': probas}
    return json.dumps(out)

def process_img(img):
    header, content = img.split(',', 2)
    with open('img.png', 'wb') as fd:
        d = base64.b64decode(content)
        fd.write(d)
    img = imread('img.png')
    img = img[:, :, 3]
    #img = autocrop(img)
    img = resize(img, (28, 28), preserve_range=True)
    img = img > threshold_otsu(img)
    img = img.astype(np.float32)
    imsave('img.png', img)
    return img

if __name__ == '__main__':
    serve()
