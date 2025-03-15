import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import base64


# Sample audio file paths (Replace with actual paths)
n15bn = "noisy-15bn.wav"
p15bn = "pred-15bn.wav"
c15bn = "clean-15bn.wav"

n15en = "noisy-15en.wav"
p15en = "pred-15en.wav"
c15en = "clean-15en.wav"
e10 = ["noisy-10en.wav", "pred-10en.wav", "clean-10en.wav"]
e5 = ["noisy-5en.wav", "pred-5en.wav", "clean-5en.wav"]

# Function to plot waveform
@st.cache_data
def plot_waveform(audio_path, title):
    y, sr = librosa.load(audio_path, sr=None)
    fig, ax = plt.subplots(figsize=(4, 1.5))  # Smaller height
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    st.pyplot(fig)
    plt.close()

# Function to plot spectrogram
@st.cache_data
def plot_spectrogram(audio_path, title):
    y, sr = librosa.load(audio_path, sr=None)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots(figsize=(4, 2))  # Small but readable
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)
    plt.close()

# Page Title
st.markdown('<h1 style="text-align:center; color:#ff4b4b;">Speech Enhancement</h1>', unsafe_allow_html=True)

# Navigation Menu
menu_options = ["Home", "Result", "Paper", "Code"]
if "page" not in st.session_state:
    st.session_state.page = "Home"

def switch_page(page):
    st.session_state.page = page

st.markdown('<div style="display: flex; justify-content: center; gap: 20px; margin-bottom: 10px;">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Home", use_container_width=True): switch_page("Home")
with col2:
    if st.button("Result", use_container_width=True): switch_page("Result")
with col3:
    if st.button("Paper", use_container_width=True): switch_page("Paper")
with col4:
    if st.button("Code", use_container_width=True): switch_page("Code")

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.page == "Home":
    st.markdown('<h3 style="text-align: center;justify-content:center;">CAR-UNet: A ConvNeXT and Attention Aided Residual UNet-based Deep Learning Model for Monaural Speech Enhancement</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size:16px; font-weight:bold; color:#ffaaff; ">Authors</div>', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; font-size: 14px;">
        Angkon Deb<sup>1</sup>, Rudra Roy<sup>2</sup>, Celia Shahnaz<sup>3</sup>, Wei-ping Zhu<sup>4</sup>, and M. Omair Ahmad<sup>5</sup>  
        <br><sup>1, 2, 3</sup> Department of EEE, BUET, Dhaka, Bangladesh  
        <br><sup>4, 5</sup> Department of ECE, Concordia University, Montreal, Canada
        </div>
    """, unsafe_allow_html=True)

# Result Page Layout
elif st.session_state.page == "Result":
    st.subheader("Speech Enhancement Results")
    st.divider()
    st.markdown('<p style="color:#ff4b4b;text-align: center"><b>-15dB Input(English-Librispeech)</b></p>', unsafe_allow_html=True)
    st.divider()

    # Layout: 3 Columns for Noisy | Predicted | Clean
    col1, col2, col3 = st.columns(3)

    # Audio Players (First Row)
    with col1:
        st.markdown("**Noisy Speech**")
        st.audio(n15en, format="audio/wav")

    with col2:
        st.markdown("**Predicted Speech**")
        st.audio(p15en, format="audio/wav")

    with col3:
        st.markdown("**Target Speech**")
        st.audio(c15en, format="audio/wav")

    # Waveform Plots (Second Row)
    col1, col2, col3 = st.columns(3)

    with col1:
        plot_waveform(n15en, "Noisy Speech Waveform")

    with col2:
        plot_waveform(p15en, "Predicted Speech Waveform")

    with col3:
        plot_waveform(c15en, "Target Speech Waveform")

    # Spectrograms (Third Row)
    col1, col2, col3 = st.columns(3)

    with col1:
        plot_spectrogram(n15en, "Noisy Speech Spectrogram")

    with col2:
        plot_spectrogram(p15en, "Predicted Speech Spectrogram")

    with col3:
        plot_spectrogram(c15en, "Target Speech Spectrogram")

    st.divider()
    st.markdown('<p style="color:#ff4b4b;text-align: center"><b>-15dB Input(Bengali-SUBAK.KO)</b></p>', unsafe_allow_html=True)
    st.divider()
    # Audio Players (First Row)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Noisy Speech**")
        st.audio(n15bn, format="audio/wav")

    with col2:
        st.markdown("**Predicted Speech**")
        st.audio(p15bn, format="audio/wav")

    with col3:
        st.markdown("**Target Speech**")
        st.audio(c15bn, format="audio/wav")


    col1, col2, col3 = st.columns(3)
    with col1:
        plot_waveform(n15bn, "Noisy Speech Waveform")

    with col2:
        plot_waveform(p15bn, "Predicted Speech Waveform")

    with col3:
        plot_waveform(c15bn, "Target Speech Waveform") 
    # Spectrograms (Third Row)
    col1, col2, col3 = st.columns(3)

    with col1:
        plot_spectrogram(n15bn, "Noisy Speech Spectrogram")

    with col2:
        plot_spectrogram(p15bn, "Predicted Speech Spectrogram")

    with col3:
        plot_spectrogram(c15bn, "Target Speech Spectrogram")

    st.divider()
    st.markdown('<p style="color:#ff4b4b;text-align: center"><b>-10dB Input(English-LibriSpeech)</b></p>', unsafe_allow_html=True)
    st.divider()
    # Audio Players (First Row)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Noisy Speech**")
        st.audio(e10[0], format="audio/wav")

    with col2:
        st.markdown("**Predicted Speech**")
        st.audio(e10[1], format="audio/wav")

    with col3:
        st.markdown("**Target Speech**")
        st.audio(e10[2], format="audio/wav")


    col1, col2, col3 = st.columns(3)
    with col1:
        plot_waveform(e10[0], "Noisy Speech Waveform")

    with col2:
        plot_waveform(e5[1], "Predicted Speech Waveform")

    with col3:
        plot_waveform(e10[2], "Target Speech Waveform") 
    # Spectrograms (Third Row)
    col1, col2, col3 = st.columns(3)

    with col1:
        plot_spectrogram(e10[0], "Noisy Speech Spectrogram")

    with col2:
        plot_spectrogram(e5[1], "Predicted Speech Spectrogram")

    with col3:
        plot_spectrogram(e10[1], "Target Speech Spectrogram")
    

    st.divider()
    st.markdown('<p style="color:#ff4b4b;text-align: center"><b>-5dB Input(English-LibriSpeech)</b></p>', unsafe_allow_html=True)
    st.divider()
    # Audio Players (First Row)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Noisy Speech**")
        st.audio(e5[0], format="audio/wav")
    with col2:
        st.markdown("**Predicted Speech**")
        st.audio(e5[1], format="audio/wav")
    with col3:
        st.markdown("**Target Speech**")
        st.audio(e5[2], format="audio/wav")
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_waveform(e5[0], "Noisy Speech Waveform")
    with col2:
        plot_waveform(e10[1], "Predicted Speech Waveform")
    with col3:
        plot_waveform(e5[2], "Target Speech Waveform")
    # Spectrograms (Third Row)
    col1, col2, col3 = st.columns(3)
    with col1:
        plot_spectrogram(e5[0], "Noisy Speech Spectrogram")
    with col2:
        plot_spectrogram(e10[1], "Predicted Speech Spectrogram")
    with col3:  
        plot_spectrogram(e5[2], "Target Speech Spectrogram")

    st.divider()
    st.markdown('<p style="color:#ff4b4b;text-align: center"><b>Baseline Comparison (-15dB SNR)</b></p>', unsafe_allow_html=True)
    st.divider()
    st.image("compare.png", caption="STFT comparison between the baseline and the proposed model")
    st.markdown('''IIFCNet:  W. Wei, Y. Hu, H. Huang, and L. He, “Iifc-net: A monaural speech enhancement network with high-order information interaction and feature calibration,” IEEE Signal Processing Letters, vol. 31 pp. 196–200, 2023.''')
    st.markdown('''MN-Net: Y. Hu, Q. Yang, W. Wei, L. Lin, L. He, Z. Ou, and W. Yang, “Mn-net: Speech enhancement network via modeling the noise,” IEEE Transactions on Audio, Speech and Language Processing, 2025.''')
    st.markdown('''CMGAN: S. Abdulatif, R. Cao, and B. Yang, “Cmgan: Conformer-based metricgan for monaural speech enhancement,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2024.''') 

    st.divider()
    st.markdown('<p style="color:#ff4b4b;text-align: center"><b>Performance Matrix</b></p>', unsafe_allow_html=True)
    st.divider()
    st.image("table.png", caption="Performance matrix of the proposed model")

    st.image("comptable.png", caption="Performance matrix comparion with baseline models")
    st.image("noisetype.png", caption="PESQ score evaluation for different noise types")


    

elif st.session_state.page == "Paper":
    st.subheader("Research Paper")

    # PDF file path (Replace with actual file path)
    pdf_path = "Abstract.pdf"

    # Display the PDF with iframe
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

    # Optional: Provide Download Button
    st.download_button(label="📄 Download Paper", data=open(pdf_path, "rb"), file_name="Abstract.pdf", mime="application/pdf",type='primary',)


elif st.session_state.page == "Code":
    st.subheader("Source Code")

    # Embed the converted HTML notebook
    notebook_html = "my-notebook.html"

    with open(notebook_html, "r", encoding="utf-8") as f:
        notebook_content = f.read()

    st.components.v1.html(notebook_content, height=800, scrolling=True)
