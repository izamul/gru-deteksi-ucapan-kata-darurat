import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
import tempfile
import os
from pydub import AudioSegment
import io
import threading
import queue
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
import logging
import datetime
import matplotlib.pyplot as plt

# Configure TensorFlow
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Set page config
st.set_page_config(
    page_title="Sistem Deteksi Kata Darurat",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
TARGET_SR = 16000
N_MELS = 128
N_MFCC = 13  # Changed back to 13 as per model's expectation
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0
FMAX = None
DURATION = 3  
MIN_MFCC_FRAMES = 9
ALLOWED_EXTENSIONS = {'wav', 'm4a'}

# Create necessary directories
os.makedirs("processed", exist_ok=True)
os.makedirs("uploaded", exist_ok=True)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    .emergency-text {
        color: #FF4B4B;
        font-weight: bold;
    }
    .confidence-high {
        color: #FF4B4B;
    }
    .confidence-medium {
        color: #FFA500;
    }
    .confidence-low {
        color: #FFD700;
    }
    .metric-card {
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: var(--text-color);
        font-size: 0.9rem;
    }
    /* Custom styles for metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    /* Style for section headers */
    h3 {
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    /* Style for progress bar */
    .stProgress > div > div > div {
        background-color: #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://static.wikia.nocookie.net/deathbattle/images/6/6f/Portrait.feloniusgru.png", width=100)
    st.title("Sistem Deteksi Kata Darurat")
    st.markdown("---")
    st.markdown("### 📝 Tentang Sistem")
    st.markdown("""
    Sistem ini menggunakan model GRU untuk mendeteksi kata-kata darurat dalam ucapan.
    
    ### 🎯 Fitur Utama
    - Deteksi kata darurat secara real-time
    - Pemrosesan audio dengan dan tanpa pembersihan
    - Visualisasi audio untuk analisis
    - Dukungan berbagai format audio
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Pengaturan")
    processing_option = "Tanpa Pembersihan"
    
    st.markdown("---")
    st.markdown("### 📋 Format yang Didukung")
    st.markdown("""
    - WAV (Waveform Audio)
    - MP3 (MPEG Audio)
    - M4A (MPEG-4 Audio)
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ Persyaratan")
    st.markdown("""
    - Audio harus jelas (usahakan tidak ada noise)
    - Durasi: 1-3 detik
    - Audio harus terdiri 1 ucapan kata Bahasa Indonesia
    """)

# Main content
st.title("🚨 Sistem Deteksi Kata Darurat")
st.subheader("Implementasi GRU untuk Deteksi Ucapan Kata Darurat")

# Initialize session state
# if "history" not in st.session_state:
#     st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

def record_audio(duration=3, sample_rate=TARGET_SR):
    """Record audio for a fixed duration"""
    audio_data = []
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        audio_data.append(indata.copy())
    
    # Record audio (this is a blocking operation)
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        sd.sleep(int(duration * 1000))
    
    return np.concatenate(audio_data, axis=0)

def pad_audio_min_length(audio, sr, min_mfcc_frames=MIN_MFCC_FRAMES, n_fft=N_FFT, hop_length=HOP_LENGTH):
    min_len_samples = n_fft + hop_length * (min_mfcc_frames - 1)
    if len(audio) < min_len_samples:
        pad_amount = min_len_samples - len(audio)
        audio = np.pad(audio, (0, pad_amount), mode='constant')
    return audio

def preprocess_audio(audio_data, sr, apply_trim=True):
    try:
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sr != TARGET_SR:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=TARGET_SR)
        
        # Truncate silence conditionally
        if apply_trim:
            audio_data, _ = librosa.effects.trim(audio_data, top_db=10)
        
        # Pad if needed
        audio_data = pad_audio_min_length(audio_data, TARGET_SR)
        
        return audio_data
    except Exception as e:
        st.error(f"Error in audio preprocessing: {str(e)}")
        return None

def extract_mfcc_features(audio, sr):
    try:
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # Calculate delta and delta-delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack features
        mfcc_stack = np.stack([mfcc, delta, delta2], axis=0)
        mfcc_combined = mfcc_stack.transpose(2, 1, 0)
        
        return mfcc_combined
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def load_and_preprocess_audio(audio_data, sr=TARGET_SR):
    try:
        # Preprocess audio
        audio = preprocess_audio(audio_data, sr)
        if audio is None:
            return None
            
        # Extract features
        features = extract_mfcc_features(audio, TARGET_SR)
        if features is None:
            return None
            
        return features
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def predict_audio(features):
    try:
        # Verify input shape before prediction
        if features.shape != (1, 104, 13, 3):
            raise ValueError(f"Invalid input shape for prediction: {features.shape}. Expected (1, 104, 13, 3)")
        
        # Load model with custom_objects to handle metrics
        model = load_model('model/van_et_al.h5', compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Load label encoder classes
        label_encoder_classes = np.load('model/label_encoder_classes.npy', allow_pickle=True)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_encoder_classes
        
        # Make prediction
        prediction = model.predict(features, verbose=0)
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction) * 100
        
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def process_and_predict(audio_path, apply_cleaning=True):
    try:
        # Load audio with original sample rate for robustness in preprocessing
        y_raw, sr_raw = librosa.load(audio_path, sr=None)

        # Preprocess audio (mono, resample, trim - conditionally, pad) to get clean waveform
        y_processed = preprocess_audio(y_raw, sr_raw, apply_trim=apply_cleaning)
        if y_processed is None:
            return None, None, None

        # Extract MFCC features with delta and delta-delta
        mfcc = librosa.feature.mfcc(
            y=y_processed, 
            sr=TARGET_SR,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
        
        # Calculate delta and delta-delta
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack features
        mfcc_stack = np.stack([mfcc, delta, delta2], axis=0)
        mfcc_combined = mfcc_stack.transpose(2, 1, 0)
        
        # Pad or truncate to match model's expected input length
        target_length = 104  # Model's expected input length
        if mfcc_combined.shape[0] > target_length:
            mfcc_combined = mfcc_combined[:target_length, :, :]
        elif mfcc_combined.shape[0] < target_length:
            mfcc_combined = np.pad(mfcc_combined, ((0, target_length - mfcc_combined.shape[0]), (0, 0), (0, 0)), mode='constant')
        
        # Add batch dimension
        mfcc_combined = np.expand_dims(mfcc_combined, axis=0)
        
        # Verify the shape
        if mfcc_combined.shape != (1, 104, 13, 3):
            raise ValueError(f"Invalid input shape: {mfcc_combined.shape}. Expected (1, 104, 13, 3)")
        
        # Make prediction
        predicted_class, confidence = predict_audio(mfcc_combined)
        
        if predicted_class is not None:
            # Save processed audio
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_path = f"processed/processed_{timestamp}.wav"
            sf.write(processed_path, y_processed, TARGET_SR)
            
            return predicted_class, confidence, processed_path
            
    except Exception as e:
        st.error(f"Error processing audio and predicting: {str(e)}")
        return None, None, None

def visualize_audio(audio_path):
    """Visualize audio waveform and spectrogram"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=TARGET_SR)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot waveform
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title('Waveform')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        
        # Plot spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2)
        ax2.set_title('Spectrogram')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        
        # Add colorbar
        fig.colorbar(img, ax=ax2, format="%+2.f dB")
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error visualizing audio: {str(e)}")
        return None

def display_detection_results(predicted_class, confidence, processed_path):
    """Display detection results with enhanced visualization"""
    # Determine confidence level and color
    if confidence > 80:
        confidence_level = "Tinggi"
        confidence_class = "confidence-high"
    elif confidence > 60:
        confidence_level = "Sedang"
        confidence_class = "confidence-medium"
    else:
        confidence_level = "Rendah"
        confidence_class = "confidence-low"

    # Main results section
    st.markdown("### Hasil Deteksi")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Kata Terdeteksi",
            value=predicted_class,
            delta=None
        )
    
    with col2:
        st.metric(
            label="Tingkat Kepercayaan",
            value=f"{confidence:.1f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Status Tingkat Kepercayaan",
            value=confidence_level,
            delta=None
        )

    # Confidence visualization
    st.progress(confidence/100)
    
    # Audio visualization
    st.markdown("### Visualisasi Audio")
    fig = visualize_audio(processed_path)
    if fig is not None:
        st.pyplot(fig)
    
    # Audio player
    st.markdown("### Audio")
    st.audio(processed_path, format='audio/wav')

# Create two columns for recording and file upload
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎙️ Rekam Audio")
    st.markdown("Rekam suara Anda selama 3 detik untuk deteksi kata darurat.")
    
    if st.button("🎙️ Mulai Rekam", type="primary"):
        try:
            with st.spinner("🎙️ Merekam audio..."):
                audio_data = record_audio(duration=3)
            
            if audio_data is not None:
                st.success("✅ Rekaman selesai!")
                
                with st.spinner("🔄 Memproses dan menganalisis..."):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    tmp_path = f"uploaded/recorded_{timestamp}.wav"
                    sf.write(tmp_path, audio_data, TARGET_SR)
                    
                    predicted_class, confidence, processed_path = process_and_predict(tmp_path, apply_cleaning=True)
                
                if predicted_class is not None:
                    st.session_state.last_prediction = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "processed_path": processed_path
                    }
                else:
                    st.error("❌ Analisis gagal.")

            else:
                st.warning("⚠️ Tidak ada audio yang terekam.")

        except Exception as e:
            st.error(f"Error saat merekam: {str(e)}")

    if st.session_state.last_prediction is not None:
        predicted_class = st.session_state.last_prediction["class"]
        confidence = st.session_state.last_prediction["confidence"]
        processed_path = st.session_state.last_prediction["processed_path"]
        
        display_detection_results(predicted_class, confidence, processed_path)
        st.session_state.last_prediction = None

with col2:
    st.markdown("### 📤 Unggah File Audio")
    st.markdown("Unggah file audio untuk analisis kata darurat.")
    
    uploaded_file = st.file_uploader("Pilih file audio", type=list(ALLOWED_EXTENSIONS))

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            st.error(f"Format file tidak valid. Silakan unggah file {', '.join(ALLOWED_EXTENSIONS)}.")
        else:
            try:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                upload_path = f"uploaded/uploaded_{timestamp}_{uploaded_file.name}"
                with open(upload_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.audio(upload_path, format=f"audio/{file_extension}")
                st.success("✅ File berhasil diunggah!")

                with st.spinner("🔄 Memproses dan menganalisis..."):
                    predicted_class, confidence, processed_path = process_and_predict(
                        upload_path, 
                        apply_cleaning=(processing_option == "Dengan Pembersihan (Truncate Silence)")
                    )
                
                if predicted_class is not None:
                    display_detection_results(predicted_class, confidence, processed_path)
            except Exception as e:
                st.error(f"Error saat memproses file: {str(e)}")

# Footer
st.markdown("---")
st.markdown('<p style="text-align: center; color: #666;">Izamul Fikri - 2025 | Dibuat dengan ❤️ menggunakan Streamlit</p>', unsafe_allow_html=True) 