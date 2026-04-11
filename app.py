import streamlit as st
import pickle
import numpy as np
from textblob import TextBlob
import os

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="TrendLens — Viral Product Predictor",
    page_icon="🔥",
    layout="centered"
)

# ── LOAD MODEL ─────────────────────────────────────────────
@st.cache_resource
def load_model():

    BASE_DIR = os.path.dirname(__file__)

    model_path = os.path.join(BASE_DIR, "model/model.pkl")
    features_path = os.path.join(BASE_DIR, "model/features.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(features_path, "rb") as f:
        features = pickle.load(f)

    return model, features


model, features = load_model()

# ── TITLE ─────────────────────────────────────────────────
st.title("🔥 TrendLens")
st.subheader("Will this fashion product go viral?")
st.markdown("Enter product details below and our ML model will predict its viral potential.")
st.markdown("---")

# ── INPUTS ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:

    rating = st.slider(
        "⭐ Product Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.5
    )

    age = st.slider(
        "👤 Reviewer Age",
        min_value=18,
        max_value=80,
        value=35
    )

    department = st.selectbox(
        "👗 Department",
        options=[
            "Tops",
            "Dresses",
            "Bottoms",
            "Intimate",
            "Jackets",
            "Trend"
        ]
    )

with col2:

    positive_feedback = st.number_input(
        "👍 Positive Feedback Count",
        min_value=0,
        max_value=500,
        value=10
    )

    review_text = st.text_area(
        "📝 Write a Sample Review",
        placeholder="e.g. This dress is absolutely beautiful! Perfect fit and gorgeous color. Highly recommend!",
        height=120
    )

st.markdown("---")

# ── PREDICTION ───────────────────────────────────────────
if st.button("🔍 Predict Viral Potential", use_container_width=True):

    if not review_text.strip():
        st.warning("⚠️ Please write a sample review to continue.")

    else:

        # Sentiment analysis
        blob = TextBlob(review_text)
        sentiment_score = round(blob.sentiment.polarity, 4)

        if sentiment_score > 0.1:
            sentiment_encoded = 1
            sentiment_label = "😊 Positive"
        elif sentiment_score < -0.1:
            sentiment_encoded = -1
            sentiment_label = "😞 Negative"
        else:
            sentiment_encoded = 0
            sentiment_label = "😐 Neutral"

        # Department encoding
        dept_map = {
            "Tops": 4,
            "Dresses": 1,
            "Bottoms": 0,
            "Intimate": 2,
            "Jackets": 3,
            "Trend": 5
        }

        department_encoded = dept_map[department]

        # Feature calculations
        review_length = len(review_text.split())

        popularity_score = min(positive_feedback / 122, 1.0)

        # Age group encoding
        if age <= 25:
            age_group_encoded = 1
        elif age <= 35:
            age_group_encoded = 2
        elif age <= 45:
            age_group_encoded = 3
        elif age <= 60:
            age_group_encoded = 4
        else:
            age_group_encoded = 5

        high_feedback = 1 if positive_feedback > 4 else 0

        # Neutral placeholders
        class_encoded = 3
        division_encoded = 1

        # Build feature array
        input_data = np.array([[
            rating,
            age,
            sentiment_score,
            sentiment_encoded,
            popularity_score,
            review_length,
            department_encoded,
            class_encoded,
            division_encoded,
            age_group_encoded,
            high_feedback
        ]])

        # Model prediction
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data)[0][prediction] * 100

        # ── RESULTS ─────────────────────────────────
        st.markdown("## 📊 Prediction Result")

        if prediction == 1:
            st.success(f"✅ HIGH Viral Potential — {round(confidence,1)}% confidence")
            st.balloons()
        else:
            st.error(f"❌ LOW Viral Potential — {round(confidence,1)}% confidence")

        # Input summary
        st.markdown("### 🔎 Input Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Rating", f"⭐ {rating}/5")
            st.metric("Department", department)

        with c2:
            st.metric("Sentiment", sentiment_label)
            st.metric("Sentiment Score", round(sentiment_score,3))

        with c3:
            st.metric("Review Length", f"{review_length} words")
            st.metric("Positive Feedback", positive_feedback)

        st.markdown("---")

        st.markdown("### 💡 What this means")

        if prediction == 1:

            if rating >= 4.5:
                st.info("🌟 High rating + positive reviews = strong viral signal. This product has excellent chances of trending.")

            elif sentiment_encoded == 1:
                st.info("😊 Positive customer sentiment is a key driver of virality.")

            else:
                st.info("📈 Multiple signals suggest strong customer interest.")

        else:

            if rating <= 2.5:
                st.warning("⭐ Low rating reduces the chance of this product going viral.")

            elif sentiment_encoded == -1:
                st.warning("😞 Negative reviews are hurting product popularity.")

            else:
                st.warning("📉 This product lacks strong signals for virality.")

# ── FOOTER ───────────────────────────────────────────────
st.markdown("---")

st.markdown(
"""
<div style='text-align:center; color:gray; font-size:0.85em;'>

🔥 TrendLens — Built with Python, Scikit-learn & Streamlit  

Trained on **22,628 Women's Fashion Reviews**  
Random Forest Model | **93% Accuracy**

</div>
""",
unsafe_allow_html=True
)
