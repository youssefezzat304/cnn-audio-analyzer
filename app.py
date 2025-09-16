import gradio as gr
import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import librosa
from model import AudioCNN


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

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float()
        waveform = waveform.unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
audio_processor = None
classes = None


def load_model():
    """Loads the trained model and class labels from the checkpoint file."""
    global model, audio_processor, classes
    # Ensure the model is loaded only once
    if model is None:
        print("Loading model for the first time...")
        checkpoint = torch.load('best_model.pth', map_location=device)
        classes = checkpoint['classes']
        model = AudioCNN(num_classes=len(classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        audio_processor = AudioProcessor()
        print("Model loaded successfully!")


def classify_audio(audio):
    """
    Preprocesses the audio from Gradio, runs inference, and returns the results.
    """
    if audio is None:
        return {"error": "Please upload an audio file."}

    sample_rate, audio_data = audio

    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / 32768.0

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    target_sr = 44100
    if sample_rate != target_sr:
        audio_data = librosa.resample(
            y=audio_data, orig_sr=sample_rate, target_sr=target_sr)

    spectrogram = audio_processor.process_audio_chunk(audio_data)
    spectrogram = spectrogram.to(device)

    with torch.no_grad():
        output, feature_maps = model(spectrogram, return_feature_maps=True)

        output = torch.nan_to_num(output)
        probabilities = torch.softmax(output, dim=1)
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)

        predictions = [
            {"class": classes[idx.item()], "confidence": float(prob.item())}
            for prob, idx in zip(top3_probs, top3_indices)
        ]

    viz_data = {}
    for name, tensor in feature_maps.items():
        if tensor.dim() == 4:
            aggregated_tensor = torch.mean(tensor, dim=1)
            squeezed_tensor = aggregated_tensor.squeeze(0)
            numpy_array = squeezed_tensor.cpu().numpy()
            clean_array = np.nan_to_num(numpy_array)
            viz_data[name] = {
                "shape": list(clean_array.shape),
                "values": clean_array.tolist()
            }

    spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
    clean_spectrogram = np.nan_to_num(spectrogram_np)

    max_samples = 8000
    if len(audio_data) > max_samples:
        step = len(audio_data) // max_samples
        waveform_data = audio_data[::step]
    else:
        waveform_data = audio_data

    response = {
        "predictions": predictions,
        "visualization": viz_data,
        "input_spectrogram": {
            "shape": list(clean_spectrogram.shape),
            "values": clean_spectrogram.tolist()
        },
        "waveform": {
            "values": waveform_data.tolist(),
            "sample_rate": target_sr,
            "duration": len(audio_data) / target_sr
        }
    }

    return response


# Load the model when the script starts
load_model()

# --- Gradio Interface ---
with gr.Blocks(title="Audio CNN Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Audio Classification Demo
        Upload an audio file to classify it into one of the 50 ESC-50 environmental sound classes.
        The model will return the top 3 predictions with their confidence scores.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload"], type="numpy", label="Upload Audio File")
            predict_btn = gr.Button("Classify", variant="primary")
        with gr.Column(scale=2):
            api_output = gr.JSON(label="Prediction Results")

    predict_btn.click(classify_audio, inputs=audio_input, outputs=[api_output])

    gr.Examples(
        examples=[
            ["./examples/dog.wav"],
            ["./examples/chainsaw.wav"],
            ["./examples/crackling_fire.wav"],
        ],
        inputs=audio_input
    )


if __name__ == "__main__":
    demo.launch()
