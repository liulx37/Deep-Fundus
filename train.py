import os
from keras.applications import InceptionResNetV2
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from focal_loss import focal_loss
import argparse
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description = '')
parser.add_argument('--aspect', default='acceptable', choices=['acceptable', 'blur', 'blur_od',
                                                               'blur_macula', 'blur_others', 'illuminate',
                                                               'illuminate_od', 'illuminate_macula', 'illuminate_others',
                                                               'structure', 'structure_od', 'structure_macula', 'cataract'], help='quality aspect')
parser.add_argument('--image_dir', default='images', help='directory of images')
parser.add_argument('--train_file_path', default='train.csv', help='train file path in csv format')
parser.add_argument('--val_file_path', default='val.csv', help='validation file path in csv format')
args = parser.parse_args()

y_col = args.aspect

df_train = pd.read_csv(args.train_file_path).dropna(subset=[y_col])[:100]
df_train[y_col]=df_train[y_col].astype('str')[:100]
df_val = pd.read_csv(args.val_file_path).dropna(subset=[y_col])[:100]
df_val[y_col] = df_val[y_col].astype('str')[:100]

# if not os.path.exists(f"mode/{y_col}"): os.mkdir(f"model/{y_col}")

idg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    height_shift_range=0.02,
    width_shift_range=0.02,
    #brightness_range=(0.5, 1.5),
    #channel_shift_range=0.5,
)
tdg = ImageDataGenerator(rescale=1./255)

IMG_SIZE = 512
save_path = "saved_models/{}_".format(y_col)
model_checkpoints = ModelCheckpoint(save_path + "{epoch:03d}_{val_loss:.4f}_{val_accuracy:.4f}.hdf5", monitor="val_loss", save_best_only=True)

base = InceptionResNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet", pooling="avg")
x = base.output
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=x)
model.compile("adam", loss=focal_loss(alpha=0.5), metrics=["accuracy"])
model.fit_generator(
    idg.flow_from_dataframe(df_train,
        args.image_dir,
        y_col=y_col,
        class_mode="binary",
        seed=42,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=10,
    ),
    steps_per_epoch=890,
    epochs=500,
    validation_data=tdg.flow_from_dataframe(df_val,
        args.image_dir,
        y_col=y_col,
        class_mode="binary",
        seed=42,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=10,
    ),
    validation_steps=191,
    callbacks=[model_checkpoints, EarlyStopping(patience=120)],
)
