from keras.models import load_model
import flask
import numpy as numpy

app = flask.Flask(__name__)
model = None
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

def load():
    global model
    model = load_model("mnist.h5")

@app.route("/", methods = ["GET"])
def help():
    # ユーザ一覧からレスポンスを作る
    response = {'help': None}
    if flask.request.method == "GET":
        msg = """This application is for calassifying digits. You can send a request with the image having digits and then get a probability showing the digits you send are each digits."""
        response["help"] = msg
    return flask.jsonify(response)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# @app.route("/", methods=["POST"])
# def predict():
#     response = {"Content-Type": "application/json",
#                 "result": None, 
#                 "probability": None}
#     if flask.request.method == "POST":
#         if flask.request.files["file"] and allowed_file(flask.request.files["file"]):   #exist file
#             im = np.array(Image.open(flask.request.files["file"]))
#             return im
#     # return flask.request.files["file"]
#     return flask.jsonify(response)
            


if __name__ == '__main__':
    load()
    app.run()