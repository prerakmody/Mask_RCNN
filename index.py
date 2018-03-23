## WEBSERVER
from flask import Flask
from flask import abort
from flask import request
from flask import jsonify
app = Flask(__name__)

## STANDARD PYTHON LIBS
import os, sys
import json

## SETUP
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# ROOT_DIR = '/home/play/playment/production/Mask_RCNN'
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

## OUR CUSTOM LIBS
import src.predict as predict

######################################
#            SAMPLE PAGE             #
######################################
@app.route('/')
def index():
    # ip = os.popen('curl ipecho.net/plain').read()
    base_url = 's3.amazonaws.com/open_datasets/open_datasets/mapillary/mapillary-vistas-dataset_public_v1.0/training/images/'
    sample_images = [
	  '-4Kw4AJqOzZG8q5le7bGlQ.jpg'
	, '-4SR45KuNoLQJJMNstPwRg.jpg'
	]
    block = """
	<a href='/img_url/{0}'>Mask-R-CNN Predict1</a><br />
        <a href='/img_url/{1}'>Mask-R-CNN Predict2</a>
    """.format(
	os.path.join(base_url, sample_images[0])
	, os.path.join(base_url, sample_images[1])
    )
    return block

######################################
#            API ROUTES              #
######################################
@app.route('/predict', methods=['POST'])
def get_masks():
    flag_formdata = 0
    if request.method == 'POST':
        data = json.loads(request.get_data().decode())
        for key in data:
            if 'img_url' == key:
                flag_formdata = 1
                img_url = data[key]
                res = predict.predict(img_url, model, config)
                res = jsonify(res)
                return res

        if flag_formdata == 0:
            abort(404)

######################################
#            APP RUN                 #
######################################
if __name__ == "__main__":
    print ('\n================================================\n')
    global model, config
    model, config = predict.load_model(ROOT_DIR)
    # model = predict.load_model(ROOT_DIR, device = 'cpu')
    if model != []:
        print ('\n================================================\n')
        app.run(host='0.0.0.0', port=5000)
