from flask import Flask,jsonify,render_template,request
import predict
import numpy as np

app = Flask(__name__)

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

    test = np.zeros((predict.seq_len, predict.input_size))
    for i in range(predict.seq_len):
        test[i] = data[:,i]

    result = predict.main(test)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
