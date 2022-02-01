# Imports : 
import os
os.system('apt install git')
os.system('apt install libgl1-mesa-glx -y')
os.system('pip install -q git+https://github.com/tensorflow/examples.git')
os.system('pip install opencv-python')
from azureml.core import Run
from azureml.core import Dataset
import glob
import cv2
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout, Reshape, Permute
from tensorflow.keras.optimizers import Adadelta, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, binary_crossentropy
import tqdm
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
print('Import packages OK')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Création du générateur avec augmentation d'images : 
class SegmentationGenerator(Sequence):
    
    def __init__(self, imgpaths, maskpaths, mode='train', n_classes=8, batch_size=1, resize_shape=(224,224), 
                 seed = 7, crop_shape=None, horizontal_flip=True, blur = 0,
                 vertical_flip=0, brightness=0.1, rotation=5.0, zoom=0.1, do_ahisteq = True):        
        self.blur = blur
        self.histeq = do_ahisteq
        self.image_path_list = imgpaths
        self.label_path_list = maskpaths
        # np.random.seed(seed)      
        self.mode = mode
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness = brightness
        self.rotation = rotation
        self.zoom = zoom
        self.Y=None
        # Preallocate memory
        if self.crop_shape:
            self.X = np.zeros((batch_size, crop_shape[1], crop_shape[0], 3), dtype='float32')
            self.Y = np.zeros((batch_size, crop_shape[1], crop_shape[0], self.n_classes), dtype='float32')
        elif self.resize_shape:
            self.X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.Y = np.zeros((batch_size, resize_shape[0],resize_shape[1], self.n_classes), dtype='float32')
        else:
            raise Exception('No image dimensions specified!')
        
    def __len__(self):
        return len(self.image_path_list) // self.batch_size
        
    def __getitem__(self, i):
        
        for n, (image_path, label_path) in enumerate(zip(self.image_path_list[i*self.batch_size:(i+1)*self.batch_size], 
                                                        self.label_path_list[i*self.batch_size:(i+1)*self.batch_size])):
            
            image = cv2.imread(image_path, 1)
            label = cv2.imread(label_path, 0)
            labels = np.unique(label)
            label = np.squeeze(label)
            mask = np.zeros((label.shape[0], label.shape[1], self.n_classes))
            for i in range(-1, 34):
                for p,q in enumerate(cats.keys()):
                    if i in cats[q]:
                        mask[:,:,p] = np.logical_or(mask[:,:,p],(label==i)) 
            label = np.resize(mask,(image.shape[0], image.shape[1], self.n_classes))
            
            if self.blur and random.randint(0,1):
                image = cv2.GaussianBlur(image, (self.blur, self.blur), 0)
                

            if self.resize_shape and not self.crop_shape:
                image = cv2.resize(image, self.resize_shape)
                label = cv2.resize(label, self.resize_shape, interpolation = cv2.INTER_NEAREST)
        
            if self.crop_shape:
                image, label = _random_crop(image, label, self.crop_shape)

            if self.horizontal_flip and random.randint(0,1):
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)
            
            if self.vertical_flip and random.randint(0,1):
                image = cv2.flip(image, 0)
                label = cv2.flip(label, 0)
            
            if self.brightness:
                factor = 1.0 + random.gauss(mu=0.0, sigma=self.brightness)
                if random.randint(0,1):
                    factor = 1.0/factor
                table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
                image = cv2.LUT(image, table)
            
            if self.rotation:
                angle = random.gauss(mu=0.0, sigma=self.rotation)
            else:
                angle = 0.0
            
            if self.zoom:
                scale = random.gauss(mu=1.0, sigma=self.zoom)
            else:
                scale = 1.0
            
            if self.rotation or self.zoom:
                M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, scale)
                image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]))
            
            if self.histeq: # and convert to RGB
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
                image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # to BGR
            
            self.Y[n]  = label # np.expand_dims(y, -1)
            self.X[n] = image
            
        return self.X, self.Y  #, sample_dict
print('Generator class OK')

# Création des métriques et des fonctions de perte : 

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
print('metrics OK')

# Création du Unet avec déconvolution 'à trous' : 

def conv_block(tensor, nfilters, size=3, padding='same', initializer="glorot_uniform"):
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def Unet(img_height, img_width, nclasses=8, filters=64):
# down
    input_layer = Input(shape=(img_height, img_width, 3), name='image_input')
    conv1 = conv_block(input_layer, nfilters=filters)
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(conv1_out, nfilters=filters*2)
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(conv2_out, nfilters=filters*4)
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(conv3_out, nfilters=filters*8)
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4_out = Dropout(0.3)(conv4_out)
    conv5 = conv_block(conv4_out, nfilters=filters*16)
    conv5 = Dropout(0.3)(conv5)
# up
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters*8)
    deconv6 = Dropout(0.3)(deconv6)
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters*4)
    deconv7 = Dropout(0.3)(deconv7) 
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters*2)
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)
# output
    output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1), kernel_initializer='glorot_uniform')(deconv9)
    output_layer = BatchNormalization()(output_layer)
#     output_layer = Reshape((img_height*img_width, nclasses), input_shape=(img_height, img_width, nclasses))(output_layer)
    output_layer = Activation('softmax')(output_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')
    return model   

# Arguments d'entrée du script : 
parser = argparse.ArgumentParser()
parser.add_argument('--workspace', dest='ws', help='azure workspace')
parser.add_argument('--datafolder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--epochs', type=int, dest='epochs', default=1, help='training epochs')
args = parser.parse_args()
data_folder = args.data_folder
ws = args.ws
epochs = args.epochs


# Création et compilation du modèle : 
run = Run.get_context()

# Chargement des chemins de fichiers d'entraînement et de validation
train_img_ids = {}
val_img_ids = {}
train_image_list = []
val_image_list = []
train_mask_list = []
val_mask_list = []
ref_mask_list = []
for city in os.listdir(data_folder + '/leftImg8bit/train'): 
    train_img_dir = data_folder + '/leftImg8bit/train/' + city + "/"
    train_img_ids[city] = [n[8:-16] for n in os.listdir(train_img_dir)]
    for i in os.listdir(train_img_dir):
        train_image_list.append(train_img_dir + i)
    train_mask_dir = data_folder + '/gtFine/train/' + city + "/"
    for i in os.listdir(train_mask_dir):
        if "labelIds.png" in i:
            train_mask_list.append(train_mask_dir + i)
        elif "color.png" in i:
            ref_mask_list.append(train_mask_dir + i)

for city in os.listdir(data_folder + '/leftImg8bit/val'):     
    val_img_dir = data_folder + '/leftImg8bit/val/' + city + "/"
    val_img_ids[city] = [n[8:-16] for n in os.listdir(val_img_dir)]
    for i in os.listdir(val_img_dir):
        val_image_list.append(val_img_dir + i)
    val_mask_dir = data_folder + '/gtFine/val/' + city + "/"
    for i in os.listdir(val_mask_dir):
        if "labelIds.png" in i : 
            val_mask_list.append(val_mask_dir + i)
print(f'. . . . .Number of train_images: {len(train_image_list)}\n. . . . .Number of train masks: {len(train_mask_list)}')
print(f'. . . . .Number of val_images: {len(val_image_list)}\n. . . . .Number of val masks: {len(val_mask_list)}')
print('Import filepaths OK')

#Adressage des catégories visuelles : 
cats = {'void': [0, 1, 2, 3, 4, 5, 6],
 'flat': [7, 8, 9, 10],
 'construction': [11, 12, 13, 14, 15, 16],
 'object': [17, 18, 19, 20],
 'nature': [21, 22],
 'sky': [23],
 'human': [24, 25],
 'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}

#Instantiation des générateurs d'entraînement(augmenté) et de validation(non augmenté): 
augmented_train_gen = SegmentationGenerator(train_image_list, train_mask_list, n_classes=8, batch_size=8, resize_shape=(224,224), 
                 seed = 7, crop_shape=None, horizontal_flip=True, blur = 0,
                 vertical_flip=False, brightness=0.1, rotation=20.0, zoom=0.1, do_ahisteq = True)
valid_gen = SegmentationGenerator(val_image_list, val_mask_list, n_classes=8, batch_size=8, resize_shape=(224,224), 
                 seed = 7, crop_shape=None, horizontal_flip=False, blur = False,
                 vertical_flip=False, brightness=False, rotation=False, zoom=False, do_ahisteq = False)
print('Generator instances OK')

# Paramètres d'entrée : 
random_idx = random.randint(0,len(train_mask_list))
sample_image = image.img_to_array(image.load_img(train_image_list[random_idx], target_size=(224,224)))
sample_mask = image.img_to_array(image.load_img(train_mask_list[random_idx], target_size=(224,224)))
reference_mask = image.img_to_array(image.load_img(ref_mask_list[random_idx], target_size=(224,224)))
batch_size = 16
samples = len(train_image_list)
steps = samples//batch_size
validation_steps = len(val_image_list)//batch_size
img_height, img_width = 224, 224
classes = 8
filters_n = 64

unet = Unet(img_height, img_width, nclasses=8, filters=filters_n)
unet.compile(optimizer='adam',
              loss=combined_loss,
              metrics=[dice_coeff, 'accuracy'])
print('Model instance OK')

# Définition des callbacks et entraînement : 
tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(mode='max',
                     filepath='outputs/checkpoint',
                     monitor='val_dice_coeff',
                     save_best_only='True',
                     save_weights_only='True',
                     verbose=1)
es = EarlyStopping(mode='max', monitor='val_dice_coeff', patience=3, verbose=1)
callbacks = [tb, mc, es]

for epoch in range(epochs):
    model_history = unet.fit_generator(augmented_train_gen,
                                                  steps_per_epoch=steps,
                                                  validation_steps=validation_steps,
                                                  epochs = epoch+1,
                                                  initial_epoch=epoch,
                                                  use_multiprocessing=True,
                                                  workers=4,
                                                  validation_data=valid_gen,
                                                  callbacks=callbacks)
    run.log_list('dice_coeff', model_history.history['dice_coeff'])
    run.log_list('validation_dice_coeff', model_history.history['val_dice_coeff'])
    run.log_list('accuracy', model_history.history['accuracy'])
    run.log_list('validation_accuracy', model_history.history['val_accuracy'])
    run.log_list('loss', model_history.history['loss'])
    run.log_list('validation_loss', model_history.history['val_loss'])

# Sauvegarde du modèle dans les outputs de l'expérience en cours : 
os.makedirs('outputs', exist_ok=True)
unet.load_weights('outputs/checkpoint')
unet.save('outputs/atrous_deconv_cnn')