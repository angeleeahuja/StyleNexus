import os
import pickle
import numpy as np
import pandas as pd
from keras import regularizers
from keras.models import Model
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, Activation

def download_and_unzip_kaggle_dataset():
    os.system('kaggle datasets download -d paramaggarwal/fashion-product-images-dataset --unzip')

def load_data():
    images_df = pd.read_csv('dataset/images.csv')
    styles_df = pd.read_csv('dataset/styles.csv', on_bad_lines='skip')
    images_df['id'] = images_df['filename'].apply(lambda x: x.replace('.jpg', '')).astype(int)

    dataset = styles_df.merge(images_df, on='id', how='left').reset_index(drop=True)
    dataset['filename'] = dataset['filename'].apply(lambda x: os.path.join('fashion-product-images-dataset/fashion-dataset/images', x))
    dataset = dataset.dropna()
    return dataset

def clean_up_files(images_folder, dataset):
    all_files = os.listdir(images_folder)
    existing_filenames = dataset['filename'].apply(lambda x: os.path.basename(x)).tolist()

    for file_name in all_files:
        if file_name not in existing_filenames:
            file_path = os.path.join(images_folder, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}")

def preprocess_images(image_path, model):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        image_array = image.img_to_array(img)
        expanded_image_array = np.expand_dims(image_array, axis=0)
        preprocessed_image = preprocess_input(expanded_image_array)
        features = model.predict(preprocessed_image)
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
    
def extract_features(image_paths, model):
    features_list = []
    for path in image_paths:
        if os.path.exists(path):
            features = preprocess_images(path, model)
            if features is not None:
                features_list.append(features)
        else:
            print(f"Warning: File does not exist at path {path}")
    return np.array(features_list)

def build_model(vgg_input_shape, metadata_input_shape):
    vgg_input = Input(shape= vgg_input_shape, name='vgg_features')
    metadata_input = Input(shape= metadata_input_shape, name='metadata')
    x = Concatenate()([vgg_input, metadata_input])

    x = Dense(512, activation=None, kernel_regularizer=regularizers.L2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation=None, kernel_regularizer=regularizers.L2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation=None, kernel_regularizer=regularizers.L2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    season_x = Dense(64, activation=None, kernel_regularizer=regularizers.L2(0.01))(x)
    season_x = BatchNormalization()(season_x)
    season_x = Activation('relu')(season_x)
    season_x = Dropout(0.5)(season_x)
    season_output = Dense(4, activation='softmax', name='season_output')(season_x)

    usage_x = Dense(64, activation=None, kernel_regularizer=regularizers.L2(0.01))(x)
    usage_x = BatchNormalization()(usage_x)
    usage_x = Activation('relu')(usage_x)
    usage_x = Dropout(0.5)(usage_x)
    usage_output = Dense(8, activation='softmax', name='usage_output')(usage_x)

    gender_x = Dense(64, activation=None, kernel_regularizer=regularizers.L2(0.01))(x)
    gender_x = BatchNormalization()(gender_x)
    gender_x = Activation('relu')(gender_x)
    gender_x = Dropout(0.5)(gender_x)
    gender_output = Dense(5, activation='softmax', name='gender_output')(gender_x)

    color_x = Dense(128, activation=None, kernel_regularizer=regularizers.L2(0.01))(x)
    color_x = BatchNormalization()(color_x)
    color_x = Activation('relu')(color_x)
    color_x = Dropout(0.5)(color_x)
    color_x = Dense(64, activation=None, kernel_regularizer=regularizers.L2(0.01))(color_x)
    color_x = BatchNormalization()(color_x)
    color_x = Activation('relu')(color_x)
    color_x = Dropout(0.5)(color_x)
    color_output = Dense(46, activation='softmax', name='color_output')(color_x)

    master_category_x = Dense(64, activation=None, kernel_regularizer=regularizers.L2(0.01))(x)
    master_category_x = BatchNormalization()(master_category_x)
    master_category_x = Activation('relu')(master_category_x)
    master_category_x = Dropout(0.5)(master_category_x)
    master_category_output = Dense(7, activation='softmax', name='master_category_output')(master_category_x)

    sub_category_x = Dense(128, activation=None, kernel_regularizer=regularizers.L2(0.01))(x)
    sub_category_x = BatchNormalization()(sub_category_x)
    sub_category_x = Activation('relu')(sub_category_x)
    sub_category_x = Dropout(0.5)(sub_category_x)
    sub_category_x = Dense(64, activation=None, kernel_regularizer=regularizers.L2(0.01))(sub_category_x)
    sub_category_x = BatchNormalization()(sub_category_x)
    sub_category_x = Activation('relu')(sub_category_x)
    sub_category_x = Dropout(0.5)(sub_category_x)
    sub_category_output = Dense(45, activation='softmax', name='sub_category_output')(sub_category_x)

    article_type_x = Dense(128, activation=None, kernel_regularizer=regularizers.L2(0.01))(x)
    article_type_x = BatchNormalization()(article_type_x)
    article_type_x = Activation('relu')(article_type_x)
    article_type_x = Dropout(0.5)(article_type_x)
    article_type_x = Dense(64, activation=None, kernel_regularizer=regularizers.L2(0.01))(article_type_x)
    article_type_x = BatchNormalization()(article_type_x)
    article_type_x = Activation('relu')(article_type_x)
    article_type_x = Dropout(0.5)(article_type_x)
    article_type_output = Dense(141, activation='softmax', name='article_type_output')(article_type_x)

    model = Model(inputs=[vgg_input, metadata_input], 
                outputs=[season_output, usage_output, gender_output, color_output, master_category_output, sub_category_output, article_type_output])

    model.compile(
        optimizer='adam',
        loss={
            'season_output': 'sparse_categorical_crossentropy',
            'usage_output': 'sparse_categorical_crossentropy',
            'gender_output': 'sparse_categorical_crossentropy',
            'color_output': 'sparse_categorical_crossentropy',
            'master_category_output': 'sparse_categorical_crossentropy',
            'sub_category_output': 'sparse_categorical_crossentropy',
            'article_type_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'season_output': 'accuracy',
            'usage_output': 'accuracy',
            'gender_output': 'accuracy',
            'color_output': 'accuracy',
            'master_category_output': 'accuracy',
            'sub_category_output': 'accuracy',
            'article_type_output': 'accuracy'
        }
    )
    return model

def train_model(model, vgg_features, metadata_features, season_labels, usage_labels, gender_labels, color_labels, master_category_labels, sub_category_labels, article_type_labels):
    early_stopping = EarlyStopping(monitor='val_loss', patience= 10)
    model_checkpoint = ModelCheckpoint('model/best_model.keras', monitor='val_loss', save_best_only=True)

    history = model.fit(
        [vgg_features, metadata_features],
        {
            'season_output': season_labels,
            'usage_output': usage_labels,
            'gender_output': gender_labels,
            'color_output': color_labels,
            'master_category_output': master_category_labels,
            'sub_category_output': sub_category_labels,
            'article_type_output': article_type_labels
        },
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint]
    )
    return history

def main():
    download_and_unzip_kaggle_dataset()

    dataset = load_data()
    images_folder = 'fashion-product-images-dataset/fashion-dataset/images'
    clean_up_files(images_folder, dataset)

    base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
    vgg_model = Model(inputs=base_model.input, outputs=base_model.output)

    filenames = list(dataset['filename'])
    extracted_features = extract_features(filenames, vgg_model)
    np.save('features/extracted_features.npy', extracted_features)
    # extracted_features = np.load('features/extracted_features.npy')

    metadata = dataset.copy()
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(metadata[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']])
    encoded_metadata = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

    features_array = extracted_features.reshape(extracted_features.shape[0], -1)
    encoded_metadata_values = encoded_metadata.values
    combined_features = np.hstack([features_array, encoded_metadata_values])

    vgg_input_shape = (512,)
    metadata_input_shape = (encoded_metadata.shape[1],)
    model = build_model(vgg_input_shape, metadata_input_shape)
    
    season_encoder = LabelEncoder().fit_transform(dataset['season'])
    usage_encoder = LabelEncoder().fit_transform(dataset['usage'])
    gender_encoder = LabelEncoder().fit_transform(dataset['gender'])
    color_encoder = LabelEncoder().fit_transform(dataset['baseColour'])
    master_category_encoder = LabelEncoder().fit_transform(dataset['masterCategory'])
    sub_category_encoder = LabelEncoder().fit_transform(dataset['subCategory'])
    article_type_encoder = LabelEncoder().fit_transform(dataset['articleType'])

    encoders = {
        'season_encoder.pkl': season_encoder,
        'usage_encoder.pkl': usage_encoder,
        'gender_encoder.pkl': gender_encoder,
        'color_encoder.pkl': color_encoder,
        'master_category_encoder.pkl': master_category_encoder,
        'sub_category_encoder.pkl': sub_category_encoder,
        'article_type_encoder.pkl': article_type_encoder
    }

    for filename, encoder in encoders.items():
        with open(filename, 'wb') as file:
            pickle.dump(encoder, file)

    history = train_model(model, features_array, encoded_metadata_values, season_encoder, usage_encoder, gender_encoder, color_encoder, master_category_encoder, sub_category_encoder, article_type_encoder)

if __name__ == "__main__":
    main()