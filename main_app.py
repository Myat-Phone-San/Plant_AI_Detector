import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import pandas as pd
import os

# --- Load model ---
try:
    model = load_model('plant_disease_model.h5')
except Exception as e:
    st.error(f"မော်ဒယ်ဖိုင်ကို load မလုပ်နိုင်ပါ။ 'plant_disease_model.h5' ဖိုင်ကိုမှန်ကန်သောနေရာတွင်ထားပါ။ အချက်အလက်: {e}")
    st.stop() # Stop the app if the model can't be loaded

# --- Disease class names ---
CLASS_NAMES = ('Corn___Common_rust', 'Grape___Black_rot', 'Rice___Bacterialblight', 'Rice___Hispa')

# --- Fallback for CSV encoding and robust file loading ---
def load_csv_with_fallback(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='utf-8-sig')
    except FileNotFoundError:
        st.error(f"ဖိုင်မတွေ့ပါ: '{path}' ကိုမှန်ကန်သောနေရာတွင်ထားပါ။")
        st.stop()
    except Exception as e:
        st.error(f"{path} ကိုဖတ်ရာတွင် အမှားဖြစ်ပေါ်နေသည်: {e}")
        st.stop()

# --- Load translated CSVs ---
disease_df = load_csv_with_fallback('disease_info.csv')
supplement_df = load_csv_with_fallback('supplement_info.csv')

# --- Dictionaries for lookup in Burmese ---
# Ensure 'disease_class' column exists in disease_df
if 'disease_class' in disease_df.columns:
    disease_info = disease_df.set_index('disease_class')['protection_tips'].to_dict()
    plant_name_map = disease_df.set_index('disease_class')['plant_name'].to_dict()
else:
    st.error("'disease_class' column မပါဝင်ပါ။ disease_info.csv ဖိုင်ကိုစစ်ဆေးပါ။")
    st.stop()

# Ensure 'disease_class', 'fertilizer', 'fertilizer_image' columns exist in supplement_df
if all(col in supplement_df.columns for col in ['disease_class', 'fertilizer', 'fertilizer_image']):
    supplement_info = supplement_df.set_index('disease_class')[['fertilizer', 'fertilizer_image']].to_dict(orient='index')
else:
    st.error("supplement_info.csv ဖိုင်တွင် 'disease_class', 'fertilizer', 'fertilizer_image' column မရှိပါ။")
    st.stop()

# --- Prediction Confidence Threshold ---
# This threshold is the primary mechanism to filter out non-plant images.
# If a cat photo still gets classified as a plant with high confidence (e.g., > 0.9),
# it means your current model's architecture/training makes it confident even on
# out-of-distribution data. In such cases, the most robust solution is to
# retrain the model to include an explicit "non-plant" class.
# Experiment with this value (e.g., 0.85, 0.9, 0.95, 0.98) to find the best balance
# for YOUR specific 'plant_disease_model.h5'.
PREDICTION_CONFIDENCE_THRESHOLD = 0.85

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
            st.error("ဓာတ်ပုံဖိုင်မှားနေပါသည်။ JPG, PNG, JFIF ဖိုင်တစ်ခုသာတင်ပါ။")
        else:
            st.image(opencv_image, channels="BGR", caption="တင်ထားသောဓာတ်ပုံ")

            resized_image = cv2.resize(opencv_image, (256, 256))
            normalized_image = resized_image / 255.0
            input_image = np.expand_dims(normalized_image, axis=0)

            predictions = model.predict(input_image)
            max_prediction_probability = np.max(predictions)
            predicted_class_index = np.argmax(predictions)
            predicted_class = CLASS_NAMES[predicted_class_index]

            # --- Core logic for handling non-plant/low confidence images ---
            if max_prediction_probability < PREDICTION_CONFIDENCE_THRESHOLD:
                st.title("❌ တင်ထားသောပုံသည် ကျွန်ုပ်တို့၏ **အပင်ရောဂါစာရင်းနှင့် မကိုက်ညီပါ**။")
                st.subheader("ကျေးဇူးပြု၍ **အပင်ရွက်ဓာတ်ပုံ**ကိုသာ တင်ပေးပါ။")
                # st.markdown("**ဤစနစ်သည် အခြားပုံများ (ဥပမာ- တိရစ္ဆာန်၊ လူ၊ အရာဝတ္ထုများ) ကို ခွဲခြားစိတ်ဖြာနိုင်စွမ်းမရှိပါ။**")
                # You can uncomment the following line for debugging if you want to see the low confidence score
                # st.info(f"ခန့်မှန်းခြေ ယုံကြည်မှုအဆင့် (နိမ့်လွန်းသောကြောင့် ငြင်းပယ်သည်): {max_prediction_probability:.2f}")
            else:
                # Proceed with detailed prediction if confidence is high enough
                parts = predicted_class.split('___')
                plant_name = plant_name_map.get(predicted_class, "မသိသောအပင်")
                disease_name = parts[1] if len(parts) == 2 else "မသိသောရောဂါ"

                # Show result
                st.title(f"🩺 ဒါဟာ {plant_name} ဖြစ်ပြီး {disease_name} ရောဂါရှိနိုင်ပါတယ်။")
                st.subheader(f"🌱 **အပင်အမျိုးအမည်:** {plant_name}")
                st.info(f"ခန့်မှန်းခြေ ယုံကြည်မှုအဆင့်: {max_prediction_probability:.2f}")

                # Show protection tips
                tips = disease_info.get(predicted_class)
                if tips:
                    st.subheader("🛡 ကာကွယ်နည်းများ:")
                    st.write(tips)
                else:
                    st.warning("ဤရောဂါအတွက် ကာကွယ်နည်း အချက်အလက် မတွေ့ပါ။")

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