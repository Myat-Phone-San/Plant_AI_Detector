import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import pandas as pd
import os

# --- Load model ---
model = load_model('plant_disease_model.h5')

# --- Disease class names ---
CLASS_NAMES = ('Corn___Common_rust', 'Grape___Black_rot', 'Rice___Bacterialblight', 'Rice___Hispa')

# --- Fallback for CSV encoding ---
def load_csv_with_fallback(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='utf-8-sig')

# --- Load translated CSVs ---
disease_df = load_csv_with_fallback('disease_info.csv')
supplement_df = load_csv_with_fallback('supplement_info.csv')

# --- Dictionaries for lookup in Burmese ---
disease_info = disease_df.set_index('disease_class')['protection_tips'].to_dict()
plant_name_map = disease_df.set_index('disease_class')['plant_name'].to_dict()
supplement_info = supplement_df.set_index('disease_class')[['fertilizer', 'fertilizer_image']].to_dict(orient='index')

# --- UI in Burmese ---
st.title("🌿 အပင်ရောဂါခန့်မှန်းစနစ်")
st.markdown("အပင်ရွက်ရဲ့ ဓာတ်ပုံတင်ပေးပါ (JPG, PNG, JFIF ဖိုင်များ).")

plant_image = st.file_uploader("ဓာတ်ပုံတစ်ခုရွေးပါ...", type=["jpg", "png", "jfif"])
submit = st.button('ရောဂါခန့်မှန်းမည်')

if submit:
    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if opencv_image is None:
            st.error("ဓာတ်ပုံဖိုင်မှားနေပါသည်။ ကျေးဇူးပြု၍ JPG, PNG, JFIF ဖိုင်တစ်ခုတင်ပါ။")
        else:
            st.image(opencv_image, channels="BGR", caption="တင်ထားသောဓာတ်ပုံ")
            st.write(f"ဓာတ်ပုံအရွယ်အစား: {opencv_image.shape}")

            resized_image = cv2.resize(opencv_image, (256, 256))
            normalized_image = resized_image / 255.0
            input_image = np.expand_dims(normalized_image, axis=0)

            predictions = model.predict(input_image)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]

            # Extract plant & disease names
            parts = predicted_class.split('___')
            plant_name = plant_name_map.get(predicted_class, "မသိသောအပင်")
            disease_name = parts[1] if len(parts) == 2 else "မသိသောရောဂါ"

            # Show result
            st.title(f"🩺 ဒါဟာ {plant_name} ဖြစ်ပြီး {disease_name} ရောဂါရှိနိုင်ပါတယ်။")
            st.subheader(f"🌱 **အပင်အမျိုးအမည်:** {plant_name}")

            # Show protection tips
            tips = disease_info.get(predicted_class)
            if tips:
                st.subheader("🛡 ကာကွယ်နည်းများ:")
                st.write(tips)
            else:
                st.warning("ဤရောဂါအတွက် ကာကွယ်နည်း မတွေ့ပါ။")

            # Show fertilizer info
            supplement = supplement_info.get(predicted_class)
            if supplement:
                st.subheader("💊 အကြံပြု အာဟာရမြှင့်ထည့်ပစ္စည်း:")
                st.write(supplement['fertilizer'])

                fert_img_path = supplement['fertilizer_image']
                if os.path.exists(fert_img_path):
                    st.image(fert_img_path, caption="အာဟာရပစ္စည်းဓာတ်ပုံ", width=300)
                else:
                    st.warning(f"ဓာတ်ပုံ မတွေ့ပါ: {fert_img_path}")
            else:
                st.warning("ဤရောဂါအတွက် အာဟာရပစ္စည်းအချက်အလက် မရှိပါ။")
    else:
        st.warning("ဓာတ်ပုံတင်ပြီးမှ ခန့်မှန်းပါ။")
