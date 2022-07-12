from flask import Flask,request,jsonify,render_template
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
    data = np.vstack(x,y,z)
    # result = predict.main(data)
    return jsonify({'result':data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)