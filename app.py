import streamlit as st
import numpy as np
import os
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model, Model
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
import pickle
from recommender import extract_features

@st.cache_resource
def load_assets():
    model = load_model('model/best_model.keras')
    base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
    vgg_model = Model(inputs=base_model.input, outputs=base_model.output)

    encoders = {}
    for name in ['season', 'usage', 'gender', 'color', 'master_category', 'sub_category', 'article_type']:
        with open(f'encoders/{name}_encoder.pkl', 'rb') as f:
            encoders[f'{name}_encoder'] = pickle.load(f)
    
    return model, vgg_model, encoders

def decode_predictions(predictions, encoder):
    pred_indices = np.argmax(predictions, axis=1)
    return encoder.inverse_transform(pred_indices)

def make_predictions(model, image_features, metadata_features, encoders):
    try:
        if image_features.shape[1] != 512:
            image_features = image_features.reshape((image_features.shape[0], 512))
        
        predictions = model.predict([image_features, metadata_features])
        if len(predictions) != 7:
            raise ValueError("Unexpected number of prediction outputs.")

        season_pred, usage_pred, gender_pred, color_pred, master_pred, sub_pred, article_pred = predictions

        decoded_season = decode_predictions(season_pred, encoders['season_encoder'])
        decoded_usage = decode_predictions(usage_pred, encoders['usage_encoder'])
        decoded_gender = decode_predictions(gender_pred, encoders['gender_encoder'])
        decoded_color = decode_predictions(color_pred, encoders['color_encoder'])
        decoded_master_category = decode_predictions(master_pred, encoders['master_category_encoder'])
        decoded_sub_category = decode_predictions(sub_pred, encoders['sub_category_encoder'])
        decoded_article_type = decode_predictions(article_pred, encoders['article_type_encoder'])

        return decoded_season[0], decoded_usage[0], decoded_gender[0], decoded_color[0], decoded_master_category[0], decoded_sub_category[0], decoded_article_type[0]
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Unknown", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"

def filter_items(data, season, usage):
    return data[(data['season'] == season) & (data['usage'] == usage)]

def recommend_outfit(data):
    if data.empty:
        return pd.DataFrame()  

    topwear = data[data['subCategory'] == 'Topwear'].sample(1)
    bottomwear = data[data['subCategory'] == 'Bottomwear'].sample(1)
    shoes = data[data['subCategory'] == 'Shoes'].sample(1)
    
    outfit = pd.concat([topwear, bottomwear, shoes], axis=0)
    if len(data[data['masterCategory'] == 'Accessories']) > 0:
        accessory = data[data['masterCategory'] == 'Accessories'].sample(1)
        outfit = pd.concat([outfit, accessory], axis=0) 
    return outfit

def visualize_outfit(outfit):
    if outfit.empty:
        st.write("No outfit available.")
        return
    
    fixed_width = 250
    fixed_height = 300

    cols = st.columns(len(outfit))
    for col, (_, item) in zip(cols, outfit.iterrows()):
        img_path = item['filepath']
        img = Image.open(img_path)
        img = img.resize((fixed_width, fixed_height))
        col.image(img, use_column_width=True)

def main():
    st.title('Style Nexus')
    model, vgg_model, encoders = load_assets()

    st.sidebar.subheader('Choose your requirements')
    seasons = ['Fall', 'Summer', 'Winter', 'Spring']
    occasions = ['Casual', 'Ethnic', 'Formal', 'Sports', 'Smart Casual', 'Travel', 'Party', 'Home']
    selected_season = st.sidebar.selectbox('Select Season', seasons)
    selected_occasion = st.sidebar.selectbox('Select Occasion', occasions)

    if st.sidebar.button('Generate Outfit'):
        images_folder = 'images'
        image_paths = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if image_paths:
            metadata = []
            for img_path in image_paths:
                try:
                    image_features = extract_features([img_path], vgg_model)
                    if image_features.shape[1] != 512:
                        image_features = image_features.reshape((image_features.shape[0], 512))
                    metadata_features = np.zeros((image_features.shape[0], 256))
                    
                    season, usage, gender, color, master, sub, article = make_predictions(model, image_features, metadata_features, encoders)
                    metadata.append({
                        'filepath': img_path,
                        'season': season,
                        'usage': usage,
                        'gender': gender,
                        'color': color,
                        'masterCategory': master,
                        'subCategory': sub,
                        'articleType': article
                    })
                except Exception as e:
                    st.error(f"Error processing image {img_path}: {e}") 
            metadata = pd.DataFrame(metadata)

            filtered_items = filter_items(metadata, selected_season, selected_occasion)
            outfit = recommend_outfit(filtered_items)
            st.write("Recommended Outfit:")
            visualize_outfit(outfit)
        else:
            st.write("No images found in the specified folder.")

if __name__ == "__main__":
    main()