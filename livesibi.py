import streamlit as st
import os
import numpy as np
import pickle
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import deque, Counter
import time

# --- CONFIGURATION ---
DATA_PATH = r"D:\kuliah coyy\Projek Butkamf\SIBI_dataset"
NUM_CLASSES = 26 # Assuming A-Z
MODEL_FILE = "sibi_model.pkl"
LABELMAP_FILE = "sibi_labelmap.pkl"

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- HELPER FUNCTIONS ---
def normalize_landmarks(landmarks):
    """Normalizes hand landmarks relative to the wrist (landmark 0)."""
    landmarks = np.array(landmarks).reshape(-1, 3)
    if landmarks.shape[0] == 0:
        return np.array([]) # Return empty if no landmarks
    base = landmarks[0] # Wrist landmark
    normalized = landmarks - base
    return normalized.flatten()

def prepare_data():
    """Prepares data from the dataset, normalizing landmarks and encoding labels."""
    X, y = [], []
    st.info(f"Mencari data di: {DATA_PATH}")
    found_data = False
    for i in range(NUM_CLASSES):
        letter = chr(ord('A') + i)
        letter_dir = os.path.join(DATA_PATH, letter)
        if os.path.exists(letter_dir) and os.path.isdir(letter_dir):
            for file_name in os.listdir(letter_dir):
                if file_name.endswith(".npy"):
                    try:
                        data = np.load(os.path.join(letter_dir, file_name))
                        if data.shape[0] == 21 and data.shape[1] == 3: # Ensure 21 landmarks with x,y,z
                            X.append(normalize_landmarks(data))
                            y.append(letter)
                            found_data = True
                        else:
                            st.warning(f"File {file_name} in {letter} has unexpected shape: {data.shape}. Skipping.")
                    except Exception as e:
                        st.error(f"Error loading {file_name} in {letter}: {e}. Skipping.")
        else:
            st.warning(f"Direktori '{letter_dir}' tidak ditemukan atau bukan direktori.")

    if not found_data:
        st.error(f"Tidak ada data .npy yang ditemukan di {DATA_PATH}. Pastikan struktur folder dan file sudah benar.")
        return None, None, None, None, None

    label_map = {letter: idx for idx, letter in enumerate(sorted(set(y)))}
    encoded_y = np.array([label_map[label] for label in y])

    X_array = np.array(X)
    if X_array.shape[0] == 0:
        st.error("Setelah normalisasi, tidak ada sampel data yang valid.")
        return None, None, None, None, None

    # Ensure feature dimension matches expected (21 landmarks * 3 coords = 63 features)
    if X_array.shape[1] != 63:
        st.error(f"Dimensi fitur tidak sesuai. Diharapkan 63, tapi ditemukan {X_array.shape[1]}. Periksa fungsi normalize_landmarks.")
        return None, None, None, None, None

    st.success(f"Ditemukan {len(X)} sampel data.")
    X_train, X_test, y_train, y_test = train_test_split(X_array, encoded_y, test_size=0.2, stratify=encoded_y, random_state=42)
    return X_train, X_test, y_train, y_test, label_map

def train_model(X_train, y_train):
    """Trains a RandomForestClassifier model."""
    st.info("Melatih model RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model's accuracy."""
    st.info("Mengevaluasi model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def predict_realtime(model, label_map, typing_mode=False):
    """Performs real-time hand gesture prediction."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Kamera tidak dapat diakses. Pastikan tidak ada aplikasi lain yang menggunakan kamera.")
        return

    inv_label_map = {v: k for k, v in label_map.items()}
    buffer = deque(maxlen=15) # Increased buffer for smoother prediction
    text = ""
    last_prediction_char = None
    last_action_time = time.time()
    cooldown = 1.5 # Increased cooldown for typing mode

    stframe = st.empty() # Placeholder for the video feed

    st.session_state["stop"] = False # Initialize or reset stop state
    stop_button_placeholder = st.empty() # Placeholder for the stop button

    while cap.isOpened() and not st.session_state.get("stop", False):
        stop_button = stop_button_placeholder.button("Stop Prediksi", key="stop_prediction_button")
        if stop_button:
            st.session_state["stop"] = True
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membaca frame dari kamera. Menghentikan prediksi.")
            break

        frame = cv2.flip(frame, 1) # Mirror the frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        current_prediction_display = "Tidak Terdeteksi"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks for prediction
                landmarks_coords = []
                for lm in hand_landmarks.landmark:
                    # MediaPipe landmarks are already normalized (0 to 1) for x, y. Z is depth.
                    landmarks_coords.extend([lm.x, lm.y, lm.z]) 

                if len(landmarks_coords) == 63: # 21 landmarks * 3 coordinates (x,y,z)
                    norm_landmarks = normalize_landmarks(landmarks_coords)
                    
                    # Ensure the normalized landmarks have the correct shape for prediction
                    if norm_landmarks.shape[0] == 63:
                        try:
                            pred_encoded = model.predict([norm_landmarks])
                            current_prediction_char = inv_label_map.get(pred_encoded[0], "Tidak Dikenal")
                            buffer.append(current_prediction_char)
                        except Exception as e:
                            st.warning(f"Error during model prediction: {e}")
                            # Keep prediction as "Tidak Terdeteksi"
                    else:
                        st.warning(f"Normalized landmarks have incorrect shape: {norm_landmarks.shape}. Skipping prediction.")
                else:
                    st.warning(f"Ditemukan {len(landmarks_coords)} landmark, diharapkan 63. Pastikan deteksi tangan berfungsi dengan baik.")

        if buffer:
            most_common_in_buffer = Counter(buffer).most_common(1)[0][0]
            current_time = time.time()

            if typing_mode:
                # Only update text if cooldown has passed AND prediction is stable/new
                if current_time - last_action_time >= cooldown:
                    if most_common_in_buffer != last_prediction_char: # Only act if new distinct prediction
                        if most_common_in_buffer == 'Z': # Space
                            text += ' '
                        elif most_common_in_buffer == 'X': # Backspace
                            text = text[:-1]
                        else:
                            text += most_common_in_buffer
                        
                        last_prediction_char = most_common_in_buffer
                        last_action_time = current_time # Reset cooldown timer
                        buffer.clear() # Clear buffer after an action to prevent immediate re-typing of same char
            
            # Always display the most common prediction from buffer
            current_prediction_display = most_common_in_buffer
        
        cv2.putText(frame, f"Huruf: {current_prediction_display}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if typing_mode:
            cv2.putText(frame, f"Teks: {text}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()
    if typing_mode:
        st.success(f"Teks akhir: {text}")
    st.info("Prediksi dihentikan.")
    st.session_state["stop"] = False # Reset stop state for next run

# --- STREAMLIT UI ---
st.title("🤟 Aplikasi Pengenalan Isyarat Tangan SIBI")

menu = st.sidebar.radio("Navigasi", ["Beranda", "Latih Model", "Evaluasi", "Prediksi Real-time", "Menulis dengan Gesture"])

if menu == "Beranda":
    st.markdown("""
        ### 👋 Selamat datang!
        Aplikasi ini digunakan untuk mengenali isyarat tangan berdasarkan alfabet **SIBI (Sistem Isyarat Bahasa Indonesia)**.

        - Gunakan **Latih Model** di sidebar untuk memproses dataset Anda dan melatih model.
        - Gunakan **Evaluasi** untuk melihat akurasi model yang sudah dilatih.
        - Gunakan **Prediksi Real-time** untuk menjalankan kamera Anda dan mengenali gesture secara langsung.
        - Gunakan **Menulis dengan Gesture** untuk mengubah gesture Anda menjadi teks interaktif.

        Pastikan dataset Anda (folder `SIBI_dataset` berisi subfolder A, B, C, dll., dengan file `.npy` di dalamnya) berada di lokasi yang benar seperti yang diatur dalam `DATA_PATH`.
    """)

elif menu == "Latih Model":
    st.subheader("📊 Latih Model")
    st.write("Proses ini akan melatih model pengenalan isyarat tangan menggunakan dataset SIBI Anda.")
    if st.button("Mulai Latih Model"):
        with st.spinner("Menyiapkan data dan melatih model... Ini mungkin membutuhkan waktu beberapa saat."):
            X_train, X_test, y_train, y_test, label_map = prepare_data()
            if X_train is not None and X_test is not None:
                model = train_model(X_train, y_train)
                try:
                    with open(MODEL_FILE, "wb") as f_model, open(LABELMAP_FILE, "wb") as f_label:
                        pickle.dump(model, f_model)
                        pickle.dump(label_map, f_label)
                    st.success(f"Model berhasil dilatih dan disimpan sebagai `{MODEL_FILE}` dan `{LABELMAP_FILE}`!")
                    accuracy = evaluate_model(model, X_test, y_test)
                    st.info(f"Akurasi model pada data uji: **{accuracy:.2f}**")
                except Exception as e:
                    st.error(f"Gagal menyimpan model atau label map: {e}")
            else:
                st.error("Gagal melatih model karena masalah data. Pastikan dataset Anda lengkap dan benar.")

elif menu == "Evaluasi":
    st.subheader("📈 Evaluasi Model")
    st.write("Memuat dan mengevaluasi akurasi model yang sudah dilatih.")
    if os.path.exists(MODEL_FILE) and os.path.exists(LABELMAP_FILE):
        try:
            with st.spinner("Memuat model dan data untuk evaluasi..."):
                model = pickle.load(open(MODEL_FILE, "rb"))
                label_map = pickle.load(open(LABELMAP_FILE, "rb"))
                # Re-prepare data just for X_test, y_test
                _, X_test, _, y_test, _ = prepare_data() 
                
                if X_test is not None and y_test is not None and len(X_test) > 0:
                    accuracy = evaluate_model(model, X_test, y_test)
                    st.success(f"Akurasi model pada data uji: **{accuracy:.2f}**")
                else:
                    st.error("Tidak cukup data uji untuk mengevaluasi model. Pastikan dataset Anda tersedia.")
        except Exception as e:
            st.error(f"Gagal memuat model atau label map, atau ada masalah dengan data: {e}")
            st.warning("Coba latih ulang model jika Anda mengalami masalah ini.")
    else:
        st.warning("Model belum dilatih. Harap latih model terlebih dahulu pada menu 'Latih Model'.")

elif menu == "Prediksi Real-time":
    st.subheader("👁️ Prediksi Isyarat Tangan Real-time")
    st.write("Arahkan tangan Anda ke kamera untuk melihat prediksi isyarat SIBI secara langsung.")
    if os.path.exists(MODEL_FILE) and os.path.exists(LABELMAP_FILE):
        try:
            with st.spinner("Memuat model untuk prediksi real-time..."):
                model = pickle.load(open(MODEL_FILE, "rb"))
                label_map = pickle.load(open(LABELMAP_FILE, "rb"))
            st.success("Model berhasil dimuat. Siap untuk prediksi!")
            st.button("Mulai Prediksi", key="start_realtime_pred")
            if st.session_state.get("start_realtime_pred"): # Only run if button is clicked
                predict_realtime(model, label_map, typing_mode=False)
        except Exception as e:
            st.error(f"Gagal memuat model atau label map: {e}. Pastikan model sudah dilatih dan disimpan dengan benar.")
    else:
        st.warning("Model belum dilatih. Harap latih model terlebih dahulu pada menu 'Latih Model'.")

elif menu == "Menulis dengan Gesture":
    st.subheader("✍️ Menulis dengan Gesture")
    st.write("Gunakan isyarat tangan Anda untuk mengetik teks. 'Z' untuk spasi, 'X' untuk backspace.")
    if os.path.exists(MODEL_FILE) and os.path.exists(LABELMAP_FILE):
        try:
            with st.spinner("Memuat model untuk mode menulis..."):
                model = pickle.load(open(MODEL_FILE, "rb"))
                label_map = pickle.load(open(LABELMAP_FILE, "rb"))
            st.success("Model berhasil dimuat. Siap untuk menulis!")
            st.button("Mulai Menulis", key="start_typing_pred")
            if st.session_state.get("start_typing_pred"): # Only run if button is clicked
                predict_realtime(model, label_map, typing_mode=True)
        except Exception as e:
            st.error(f"Gagal memuat model atau label map: {e}. Pastikan model sudah dilatih dan disimpan dengan benar.")
    else:
        st.warning("Model belum dilatih. Harap latih model terlebih dahulu pada menu 'Latih Model'.")