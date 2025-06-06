import sys
print(sys.version)
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")

genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

@st.cache_resource
def load_genre_model():
    return load_model('genre_model.h5')

model = load_genre_model()

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = image.convert('RGB').resize(target_size)
    img_array = np.array(img).astype(np.float32)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

st.title("🎵 Music Genre Classifier")
st.markdown("Upload a **spectrogram image** (from your dataset) to predict its genre.")

uploaded_file = st.file_uploader("Upload Spectrogram Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Spectrogram", use_column_width=True)

    with st.spinner("Predicting genre..."):
        model_input = preprocess_image(image)
        prediction = model.predict(model_input)[0]

        predicted_class = genre_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        top_3_indices = prediction.argsort()[-3:][::-1]
        top_3 = [(genre_labels[i], prediction[i] * 100) for i in top_3_indices]

    st.success(f"🎧 **Predicted Genre:** `{predicted_class}` ({confidence:.2f}%)")

    st.subheader("🎯 Top 3 Predictions:")
    for genre, conf in top_3:
        st.markdown(f"- `{genre}`: **{conf:.2f}%**")

    st.subheader("🔎 Prediction Breakdown")
    fig, ax = plt.subplots()
    ax.barh(genre_labels, prediction * 100, color='orange')
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Genre Prediction Probabilities')
    plt.tight_layout()
    st.pyplot(fig)
