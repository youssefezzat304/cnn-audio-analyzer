import gradio as gr
import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
import librosa
from io import BytesIO
from model import AudioCNN  # Assumes model.py is in the same dir


class AudioProcessor:
    def __init__(self):
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB()
        )

    def process_audio_chunk(self, audio_data, sample_rate):
        if sample_rate != 44100:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=44100)
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)


# Global model and processor (loaded once)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
audio_processor = None
classes = None


def load_model():
    global model, audio_processor, classes
    checkpoint = torch.load('best_model.pth', map_location=device)
    classes = checkpoint['classes']
    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    audio_processor = AudioProcessor()
    print("Model loaded!")

# Inference function for Gradio


def classify_audio(audio):
    if audio is None:
        return "Upload an audio file!", {}, {}, {}

    # audio is (sample_rate: int, waveform: np.array) from Gradio
    # Swapped: first is int (sr), second is np.array
    sample_rate, audio_data = audio

    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    spectrogram = audio_processor.process_audio_chunk(
        audio_data, sample_rate)  # Pass sr second
    spectrogram = spectrogram.to(device)

    # Rest of the function stays the same...
    with torch.no_grad():
        output, feature_maps = model(spectrogram, return_feature_maps=True)
        output = torch.nan_to_num(output)
        probabilities = torch.softmax(output, dim=1)
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        predictions = [{"class": classes[idx.item()], "confidence": prob.item()}
                       for prob, idx in zip(top3_probs, top3_indices)]

    # Viz data (simplified; return as JSON strings for Gradio)
    viz_data = {}
    for name, tensor in feature_maps.items():
        if tensor.dim() == 4:
            aggregated = torch.mean(tensor, dim=1).squeeze(0).cpu().numpy()
            clean = np.nan_to_num(aggregated)
            viz_data[name] = clean.tolist()  # Or str(clean.tolist()) if needed

    spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
    clean_spectrogram = np.nan_to_num(spectrogram_np).tolist()

    # Waveform (downsample for viz if long)
    max_samples = 8000
    if len(audio_data) > max_samples:
        waveform_data = audio_data[::len(audio_data) // max_samples]
    else:
        waveform_data = audio_data
    duration = len(audio_data) / sample_rate  # Uses the correct sr now

    return (
        predictions,  # List of dicts (display as JSON)
        viz_data,    # Dict of lists
        clean_spectrogram,  # List
        {"values": waveform_data.tolist(), "duration": duration}  # Dict
    )


# Load model on startup
load_model()

# Gradio interface
with gr.Blocks(title="Audio CNN Classifier") as demo:
    gr.Markdown("# Audio Classification Demo")
    audio_input = gr.Audio(
        sources=["upload"], type="numpy", label="Upload Audio")
    predict_btn = gr.Button("Classify")

    with gr.Row():
        # Or gr.Textbox for formatted string
        pred_output = gr.JSON(label="Top 3 Predictions")
        viz_output = gr.JSON(label="Feature Maps Viz Data")
    spectrogram_output = gr.JSON(label="Input Spectrogram")
    waveform_output = gr.JSON(label="Waveform Data")

    predict_btn.click(classify_audio, inputs=audio_input, outputs=[
                      pred_output, viz_output, spectrogram_output, waveform_output])

if __name__ == "__main__":
    demo.launch()
