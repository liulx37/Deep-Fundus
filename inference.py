import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.backend import clear_session
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model
from PIL import Image
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

aspect_to_model = {'overall': 'acceptable.hdf5', 'clarity': 'blur.hdf5', 'clarity_od': 'blur_od.hdf5',
                   'clarity_macula': 'blur_macula.hdf5', 'clarity_others': 'blur_others.hdf5',
                   'illuminate': 'illuminate.hdf5', 'illuminate_od': 'illuminate_od.hdf5',
                   'illuminate_macula': 'illuminate_macula.hdf5', 'illuminate_others': 'illuminate_others.hdf5',
                   'position': 'structure.hdf5', 'position_od': 'structure_od.hdf5', 'position_macula': 'structure_macula.hdf5',
                   'cataract': 'cataract.hdf5'}

parser = argparse.ArgumentParser(description = 'configuration')
parser.add_argument('--mode',  type=int, default=1, choices=[1, 2], help='1 indicates quality classification and 2 indicates real-time guidance')
parser.add_argument('--image_dir',  default='images', help='directory of test images')
parser.add_argument('--image_size', type=int, default=512, help='image resolution (need to match the models)')
args = parser.parse_args()

if args.mode == 1:
    res_dict = dict([(k, []) for k in aspect_to_model.keys()])
    for aspect in list(aspect_to_model.keys()):
        clear_session()
        model = load_model(os.path.join('models/', aspect_to_model[aspect]), compile=False)
        all_fn = sorted(os.listdir(args.image_dir))
        for fn in all_fn:
            try:
                image = Image.open(os.path.join(args.image_dir, fn)).convert('RGB')
                image = image.resize((512, 512))
                image = np.array(image) / 255.
                image = image.reshape(1, 512, 512, -1)
                y_pred_prob = float(np.squeeze(model.predict(image)))
                res_dict[aspect].append(y_pred_prob)
            except:
                print('error when opening {}'.format(fn))
                continue
    df_proba = pd.DataFrame(res_dict, index=all_fn)
    df_pred = df_proba.applymap(lambda x: 1 if x > 0.5 else 0)
    df_pred['illuminate_od'] = df_proba['illuminate_od'].map(lambda x: 1 if x > 0.2 else 0)
    df_pred['illuminate_macula'] = df_proba['illuminate_macula'].map(lambda x: 1 if x > 0.39 else 0)
    df_pred = df_pred.drop(['cataract'], axis=1)
    df_proba = df_proba.drop(['cataract'], axis=1)
    df_proba.to_excel('results/quality_proba.xlsx')
    df_pred.to_excel('results/quality_pred.xlsx')

else:
    all_fn = sorted(os.listdir(args.image_dir))
    predictions = []
    for fn in all_fn:
        result = ''
        for aspect in ['overall', 'position', 'illuminate', 'clarity', 'cataract']:
            clear_session()
            model = load_model(os.path.join('/data/yellowcard/llx/models/', aspect_to_model[aspect]), compile=False)
            try:
                image = Image.open(os.path.join(args.image_dir, fn)).convert('RGB')
                image = image.resize((512, 512))
                image = np.array(image) / 255.
                image = image.reshape(1, 512, 512, -1)
                y_pred_prob = float(np.squeeze(model.predict(image)))

            except:
                print('error when opening {}'.format(fn))
                continue
            if aspect == 'overall' and y_pred_prob <= 0.5:
                result = 'finish'
                break
            if aspect == 'position' and y_pred_prob > 0.5:
                result = 'recapture'
                break
            if aspect == 'illuminate' and y_pred_prob > 0.5:
                result = 'recapture'
                break
            if aspect == 'clarity' and y_pred_prob <= 0.5:
                result = 'finish'
                break
            if aspect == 'cataract':
                if y_pred_prob > 0.5:
                    result = 'referral'
                else:
                    result = 'recapture'
        predictions.append(result)
    df = pd.DataFrame({'prediction': predictions}, index=all_fn)
    df.to_excel('results/advice_pred.xlsx')