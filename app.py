from keras.models import load_model
import flask
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend as K


app = flask.Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

def load():
    model = load_model("mnist.h5", compile=False)
    return model

@app.route("/", methods = ["GET"])
def help():
    response = {"Content-Type": "application/json", 'help': None}
    if flask.request.method == "GET":
        msg = 'exp. ```curl -F "file=[filename].jpg" "http://localhost:5000/"```'
        response["help"] = msg
    return flask.jsonify(response)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def transform_img(img):
    img = np.array(img).convert('L')
    width,height = 28, 28
    img = np.resize(img, (width,height))
    img = img.reshape((1, width, height, 1))
    img = img.astype('float32')
    img /= 255
    return img

@app.route("/", methods=["POST"])
def predict():
    model = load()
    response = {"Content-Type": "application/json",
                "result": None, 
                "probability": None}
    if flask.request.method == "POST":
        if flask.request.files["file"]:
            img = Image.open(flask.request.files["file"])
            img = transform_img(img)
            result = model.predict(img,verbose=0)
            K.clear_session()
            response["result"] = str(np.argmax(result))
            response["probability"] = str(np.max(result))
    return flask.jsonify(response)
            

if __name__ == '__main__':
    # load()
    app.run()