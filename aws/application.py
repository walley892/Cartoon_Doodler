from flask import Flask
from flask import request
import flask
from Classifier import Classifier
from PIL import Image
import Process
import numpy as np
import pickle


app = Flask(__name__)

@app.route('/', methods = ['GET','POST'])
def action():
    if request.method =='POST':
        #return rep(str(request.files['image']))

        clf = Classifier()
        clf.load('saved')
        f = Process.get_image_from_request(request, size = (250,250))
        pred = np.argmax(clf.predict(f)[0])
        label_dict = pickle.load(open('label_dict','rb'))
        prediction = 'rick'
        for key in label_dict:
            if label_dict[key] = pred:
                prediction = key
                break
        clf.kill()
        del f
        del pred
        return rep(str(prediction))
    else:
        return rep('helloooo!!!!')

def rep(data):
    response = flask.Response(data)
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
application = app
