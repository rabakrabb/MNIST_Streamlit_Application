import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
import cv2
from skimage.morphology import dilation, footprint_rectangle
from skimage.filters import threshold_otsu
from streamlit_drawable_canvas import st_canvas
import requests
from io import BytesIO

# Sätt upp sidan med titel, ikon och layout
st.set_page_config(page_title="Handskriven sifferigenkänning", layout="centered")


# Dropbox länk till din fil
# dropbox_link = "https://www.dropbox.com/scl/fi/jv8gjbakxndgolf23lj6w/random_forest_model_full_data.pkl?rlkey=lt1948c93gvo1ae8k3xqzz00d&st=wedisd3q&dl=1"

# Ladda modellen från Dropbox
#def load_model_from_dropbox(dropbox_url):
    #try:
#        response = requests.get(dropbox_url, timeout=10)  # Timeout efter 10 sekunder
 #       if response.status_code == 200:
  #          st.write("Modellen laddades framgångsrikt från Dropbox!")
   #         model = joblib.load(BytesIO(response.content))
    #        return model
     #   else:
      #      st.error(f"Kunde inte ladda modellen. Statuskod: {response.status_code}")
       #     return None
    #except requests.exceptions.RequestException as e:
     #   st.error(f"Ett nätverksfel inträffade: {e}")
      #  return None


# Kontrollera om modellen redan är laddad i session_state
#if 'rf_model' not in st.session_state:
 #   with st.spinner('Laddar modellen från Dropbox...'):
  #      st.session_state.rf_model = load_model_from_dropbox(dropbox_link)
#
#rf = st.session_state.rf_model

rf = joblib.load('random_forest_model_full_data.pkl')

def binarize_image(image_array):
    threshold_value = threshold_otsu(image_array)
    image_bin = image_array > threshold_value
    return image_bin.astype(np.uint8) * 255

def center_digit(image_array):
    if image_array.max() <= 1:
        image_array = (image_array * 255).astype(np.uint8)

    M = cv2.moments(image_array)
    
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 14, 14  # Standardvärde om ingen siffra hittas
    
    height, width = image_array.shape
    center_x, center_y = width // 2, height // 2
    
    shift_x = center_x - cx
    shift_y = center_y - cy
    
    M_translate = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered_image = cv2.warpAffine(image_array, M_translate, (width, height), borderValue=0)
    
    return centered_image

def preprocess_for_prediction(image_array):
    # Binarisera bilden
    image_array = binarize_image(image_array)
    
    # Centrera bilden
    image_array = center_digit(image_array)
    
    # Dilatation för att förbättra strukturen
    image_array = dilation(image_array, footprint_rectangle((2, 2)))  # Tuple
    
    # Platta ut bilden och normalisera den
    return image_array.reshape(1, -1)  # För att platta ut bilden till en 1D-array

# Streamlit-app UI
def main():
    # Lägg till anpassad CSS för att ändra länkfärgen
    st.markdown("""<style>a, a:link, a:visited { color: #FFA500 !important;  } a:hover, a:active { color: #FF4500 !important; }</style>""", unsafe_allow_html=True)

    menu = ["Hem", "Om appen"]
    st.sidebar.markdown("### Navigering")
    selection = st.sidebar.radio("Välj sida", menu)

    if selection == "Hem":
        st.markdown("<h1 class='header-title'>Handskriven sifferigenkänning</h1>", unsafe_allow_html=True)

        if 'correct_count' not in st.session_state:
            st.session_state.correct_count = 0
            st.session_state.wrong_count = 0
            st.session_state.prediction = None

        canvas_result = st_canvas(
            fill_color="#FFFFFF",
            stroke_width=15,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=200,
            width=200,
            drawing_mode="freedraw",
            key="canvas"
        )

        if canvas_result.image_data is not None:
            image = Image.fromarray((canvas_result.image_data[:, :, 0] > 0).astype(np.uint8) * 255)
            image = image.resize((28, 28), Image.Resampling.LANCZOS)
            image = ImageOps.invert(image)
            image_array = np.array(image).astype('float32')

            preprocessed_image = preprocess_for_prediction(image_array)
            st.session_state.prediction = rf.predict(preprocessed_image)[0]

        if st.session_state.prediction is not None:
            st.markdown(f"<h3 class='prediction-text'>Gissning: {st.session_state.prediction}</h3>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Rätt", use_container_width=True):
                    st.session_state.correct_count += 1
                    st.session_state.prediction = None

            with col2:
                if st.button("Fel", use_container_width=True):
                    st.session_state.wrong_count += 1
                    st.session_state.prediction = None

        st.markdown(f"<p class='stats-text'>Korrekt: {st.session_state.correct_count} &nbsp;&nbsp; Felaktig: {st.session_state.wrong_count}</p>", unsafe_allow_html=True)

    elif selection == "Om appen":
        st.title("Om appen")
        st.write("Denna app är skapad med MNIST-datasetet och tränad med hjälp av en Random Forest-modell. Den använder machine learning för att med hög precision känna igen handskrivna siffror. Skriv bara in valfri siffra i rutan och prova själv!")
        st.write("Jag som skapat appen heter Martin Blomqvist och studerar vid produktionssättningen Data Scientist på EC Utbildning med examen 2026. För mer information om mig och mitt arbete är du välkommen att gå in på:")
        st.write("www.github.com/rabakrabb")
        st.write("www.linkedin.com/in/martin-blomqvist")

if __name__ == "__main__":
    main()
