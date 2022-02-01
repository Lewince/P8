
# Install and import required libraries - certain packages had to be left out of
# requirements and installed in below sequence order for container to startup
# successfully

import os
os.system('pip install -r requirements.txt')
os.system('pip install azureml-core')
os.system('pip install jsonpickle')
os.system('pip install scikit-image')
os.system('pip install tensorflow-cpu')
os.system('pip install segmentation_models')
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace
from flask import Flask, request, Response, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.vgg16 import preprocess_input
import jsonpickle
import numpy as np
from skimage.io import imread
from io import BytesIO
import base64
import time
import gc
os.environ["SM_FRAMEWORK"] = "tf.keras"
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

# Image dimensions - per model input

img_height = 512
img_width = 512

# Instantiate app and Azure App Service components

app = Flask(__name__)

svc_pr_password = os.environ.get("AZUREML_SECURED_DIR")

svc_pr = ServicePrincipalAuthentication(
    tenant_id="ab8fc798-86d2-43ba-bf39-fef66c542661",
    service_principal_id="ed782c49-9843-41c1-9726-b9a7c29ae7ef",
    service_principal_password=svc_pr_password)

ws = Workspace(
    subscription_id="06e3d8ca-a338-4e94-9976-3a0e38e04b27",
    resource_group="P8",
    workspace_name="P8_workspace",
    auth=svc_pr
    )

# Loss functions are made available here for model loading purposes

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def combined_loss(y_true, y_pred):
    cce = CategoricalCrossentropy()
    loss = cce(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss

def jaccard_distance(y_true, y_pred, smooth=100):    
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# Categories below reflect the main classes of objects from cityscapes dataset
# This needs to be remapped if addressing a different segmentation problem

cats = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}

# Below colors were quickly chosen for making prediction visualisation clear

category_colors = [[0,0,0],
              [192,192,192],
              [255,128,0],
              [255,255,0],
              [0,255,0],
              [0,0,255],
              [255,0,0],
              [255,255,255]
              ]

# Below functions will colorize either the reference mask or predicted
# segmented image

def colorize_cs_mask(mask_img, category_colors=category_colors):
    mask = np.zeros((mask_img.shape[0], mask_img.shape[1], 8))
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(-1, 34):
        for p,q in enumerate(cats.keys()):
            if i in cats[q]:
                mask[:,:,p] = np.logical_or(mask[:,:,p],(mask_img[:,:,0]==i))
    for frame, cat in enumerate(cats.keys()) :     
        color_mask[np.where(np.argmax(mask,axis=-1) == frame)] = category_colors[frame]
    return color_mask

def colorize_predicted_mask(mask, category_colors=category_colors): 
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    for frame, cat in enumerate(cats.keys()) :     
        color_mask[np.where(np.argmax(mask,axis=-1) == frame)] = category_colors[frame]
    return color_mask

# Model instantiation

global model
model_path = Model.get_model_path('VGG16-unet', _workspace=ws)
model = tf.keras.models.load_model(model_path,
                                  custom_objects={
                                      'binary_crossentropy_plus_jaccard_loss': bce_jaccard_loss,
                                      'dice_coeff': dice_coeff,
                                      'iou_score': iou_score
                                  })

data_folder = '/csdata/cityscapes_data'

# Loading selected test set from mounted cloud storage

val_img_ids = {}
val_image_list = []
val_mask_list = []
for city in os.listdir(data_folder + '/leftImg8bit/val'):     
    val_img_dir = data_folder + '/leftImg8bit/val/' + city + "/"
    val_img_ids[city] = [n[8:-16] for n in os.listdir(val_img_dir)]
    for i in os.listdir(val_img_dir):
        if '.png' in i:
            val_image_list.append(val_img_dir + i)
    val_mask_dir = data_folder + '/gtFine/val/' + city + "/"
    for i in os.listdir(val_mask_dir):
        if "labelIds.png" in i: 
            val_mask_list.append(val_mask_dir + i)
val_mask_list.sort()
val_image_list.sort()
print(f'. . . . .Number of val_images: {len(val_image_list)}\n. . . . .Number of val masks: {len(val_mask_list)}')
print('Import filepaths OK')


# APP ROUTE DEFINITIONS

# Checkpage
@app.route('/')
def hello():
    return 'Welcome to cityscapes image segmentation webapp'

# Full api requiring image index in library as input
# Returns prediction, chosen image and reference mask
@app.route('/theapi', methods=['POST'])
def test():
        
    r = request
    image_index = r.headers['index']
    image_index = int(image_index)
    pic = image.load_img(val_image_list[image_index], target_size=(img_height,img_width))
    mask = image.load_img(val_mask_list[image_index], target_size=(img_height,img_width))
    time_in = time.perf_counter()
    img = image.img_to_array(pic)
    img = preprocess_input(img)
    y_pred = model.predict(img[tf.newaxis, ...])
    predicted_mask = colorize_predicted_mask(y_pred[0])
    
    if r.headers['alpha']:
        alpha = r.headers['alpha']
        alpha = float(alpha)
    else: 
        alpha = 0.01
    print(f'alpha : {alpha}')    
    annotated = alpha * img + ( 1 - alpha ) * predicted_mask
    time_out = time.perf_counter()
    
    annotated = image.array_to_img(annotated)
    buff = BytesIO()
    annotated.save(buff, format='BMP')
    byte_pred = buff.getvalue()
    
    buffe = BytesIO()
    mask = image.array_to_img(colorize_cs_mask(image.img_to_array(mask)))
    mask.save(buffe, format='BMP')
    byte_ref = buffe.getvalue()
    
    buffer = BytesIO()
    pic.save(buffer, format='BMP')
    byte_im = buffer.getvalue()
    
    exec_time = time_out - time_in
    response = {'message': 'Processed in {} seconds. Image size={}x{}'.format(exec_time, img.shape[1], img.shape[0]),
                'content': [byte_pred, byte_im, byte_ref]
                }
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

# Simple API : receives image, returns prediction (+received img for checking)
@app.route('/simpleapi', methods=['POST'])
def simpletest():
    r = request
    
    def bytes_to_ndarray(bytes):
        bytes_io = bytearray(bytes)
        by_im = imread(BytesIO(bytes_io))
        return by_im, np.array(by_im)
    
    time_in = time.perf_counter()
    pic, nparr = bytes_to_ndarray(r.data)
    img = image.img_to_array(pic)
    img = preprocess_input(img)
    y_pred = model.predict(img[tf.newaxis, ...])
    predicted_mask = colorize_predicted_mask(y_pred[0])
    
    alpha = 0.4
    annotated = alpha * img + ( 1 - alpha ) * predicted_mask
    
    annotated = image.array_to_img(annotated)
    buf = BytesIO()
    annotated.save(buf, format='BMP')
    byte_im = buf.getvalue()
    
    pic = image.array_to_img(pic)
    buffer = BytesIO()
    pic.save(buffer, format='BMP')
    byte_ref = buffer.getvalue()
    
    time_out = time.perf_counter()
    exec_time = time_out - time_in
    response = {'message': 'Processed in {} seconds. size={}x{}'.format(exec_time, img.shape[1], img.shape[0]),
                'annotated': byte_im, 'received_image' : byte_ref
               }
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

# Webapp using route params for image selection from library - displays the 3  
# images of interest on white page, enables control of image/mask blending
@app.route('/webapp', defaults={'image_index' : 1, 'alpha' : 0.01})
@app.route('/webapp/<int:image_index>/<float:alpha>')
def webapp(image_index, alpha):  
    def bytes_to_ndarray(bytes):
        bytes_io = bytearray(bytes)
        by_im = imread(BytesIO(bytes_io))
        return by_im, np.array(by_im)
    pic = image.load_img(val_image_list[image_index], target_size=(img_height,img_width))
    mask = image.load_img(val_mask_list[image_index], target_size=(img_height,img_width))
    time_in = time.perf_counter()
    img = image.img_to_array(pic)
    img = preprocess_input(img)
    y_pred = model.predict(img[tf.newaxis, ...])
    predicted_mask = colorize_predicted_mask(y_pred[0])
    
    print(f'alpha : {alpha}')    
    annotated = alpha * img + ( 1 - alpha ) * predicted_mask
    time_out = time.perf_counter()
    
    annotated = image.array_to_img(annotated)
    buff = BytesIO()
    annotated.save(buff, format='JPEG')
    byte_pred = base64.b64encode(buff.getvalue())
    
    mask = image.array_to_img(colorize_cs_mask(image.img_to_array(mask)))
    buffe = BytesIO()
    mask.save(buffe, format='JPEG')
    byte_ref = base64.b64encode(buffe.getvalue())
    
    buffer = BytesIO()
    pic.save(buffer, format='JPEG')
    byte_im = base64.b64encode(buffer.getvalue())
    
    exec_time = time_out - time_in

    return render_template('index.html',
                           image_index = image_index,
                           base_image=byte_im.decode('utf-8'),
                           ref_mask=byte_ref.decode('utf-8'),
                           pred_image=byte_pred.decode('utf-8'))

# Execution as main 
if __name__=='__main__':
    app.run()