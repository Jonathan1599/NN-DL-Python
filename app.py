

import socket
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pickle
import numpy as np
app = Flask(__name__)

import tensorflow as tf

sess=tf.Session()    
#First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('./model/my_test_model-1000.meta')
# saver.restore(sess,tf.train.latest_checkpoint('./model'))
# graph = tf.get_default_graph()
# X = graph.get_tensor_by_name("myinput_x:0")
# Z = graph.get_tensor_by_name("myinput_z:0")
# T = graph.get_tensor_by_name("myinput_t:0")
# feed_dict ={Z:13.0,X:17.0, T : 0}
# op_to_restore = graph.get_tensor_by_name("X_hat:0")

loaded_model = pickle.load(open("./model/svm_model.sav", 'rb'))
result = loaded_model.predict([[4.30e+01, 0.00e+00, 1.22e+02, 2.13e+02, 0.00e+00, 1.65e+02,
       0.00e+00, 2.00e-01, 0.00e+00, 0.00e+00, 1.00e+00, 0.00e+00,
       0.00e+00, 1.00e+00, 1.00e+00, 0.00e+00, 1.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00]])
print(result)

app.config['CORS_HEADERS'] = '*'

@app.route('/')
def hello():
    gen =  sess.run(op_to_restore, feed_dict)
    print(gen)
    return 'Hello, World!'

@app.route('/uci', methods = ['POST','GET'])
@cross_origin()
def getPrediction():
    data = request.get_json() 
    X = []
    for attribute in data.values():
        X.append(int(attribute))

    print(X)
    pred = loaded_model.predict([X])
    result = int(pred[0])
    print(result)
    return jsonify({'prediction': result })

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=3002, debug=True)


