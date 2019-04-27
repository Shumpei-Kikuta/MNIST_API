from keras.models import load_model
import flask
import numpy as np
from PIL import Image, ExifTags
from keras import backend as K


app = flask.Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

def load():
    """modelを読み込む"""
    model = load_model("mnist.h5", compile=False)
    return model

@app.route("/help", methods = ["GET"])
def help():
    response = {"Content-Type": "application/json", 'help': None}
    if flask.request.method == "GET":
        msg = 'exp. curl -F "file=[filename].jpg" "http://localhost:5000/predict/"'
        response["help"] = msg
    return flask.jsonify(response)

def allowed_file(filename):
    """画像型の拡張子になっているか確認する"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def transform_img(img):
    """読み込んだimageのshapeをMNISTのshape(28, 28)にする"""
    img = img.convert('L')
    width,height = 28, 28
    img = img.resize((width,height), Image.LANCZOS)
    img_array = np.asarray(img).reshape((1, width, height, 1))
    return img_array

def deal_rotation(img):
    """postした画像が回転してしまった場合，元に戻す"""
    for orientation in ExifTags.TAGS.keys() : 
                if ExifTags.TAGS[orientation]=='Orientation' : break 
            exif=dict(img._getexif().items())
            if   exif[orientation] == 3 : 
                img=img.rotate(180, expand=True)
            elif exif[orientation] == 6 : 
                img=img.rotate(270, expand=True)
            elif exif[orientation] == 8 : 
                img=img.rotate(90, expand=True)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    model = load()
    response = {"Content-Type": "application/json",
                "result": None, 
                "probability": None}
    if flask.request.method == "POST":
        if flask.request.files["file"]:
            img = Image.open(flask.request.files["file"])
            img = deal_rotation(img)
            img_array = transform_img(img)
            result = model.predict(img_array,verbose=0)
            K.clear_session()
            response["result"] = str(np.argmax(result))
            response["probability"] = str(np.max(result))
    return flask.jsonify(response)
            

if __name__ == '__main__':
    app.run()