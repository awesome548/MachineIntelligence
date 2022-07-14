from flask import Flask,jsonify,render_template,request
from flask_cors import CORS
import Predict
import Walk
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/post', methods=['POST'])
def post():
    req = request.json

    x = np.array(req["x"])
    y = np.array(req["y"])
    z = np.array(req["z"])
    alpha = np.array(req["alpha"])
    beta = np.array(req["beta"])
    gamma = np.array(req["gamma"])
    data = np.vstack([x, y, z, alpha, beta, gamma])
    print(data.shape)

    test = np.zeros((Predict.seq_len, Predict.input_size))
    for i in range(Predict.seq_len):
        test[i] = data[:,i]

    walk = Walk.main(test)
    result = Predict.main(test)

    if(walk == "Walking"):
        print(jsonify({'walk':walk}))
        print(jsonify({'result':result}))
    return jsonify({'result':result,'walk':walk})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
