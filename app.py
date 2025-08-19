import os
import torch
import numpy as np
import gradio as gr
import json
import resampy
import gdown
from model import Tacotron2
from hparams import create_hparams
from text import text_to_sequence
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser

# Download models
tacotron_id = "1--eW5nk5ijbpgBqEt1TdBPr9nopcjuHE"
hifigan_url = "https://github.com/justinjohn0306/tacotron2/releases/download/assets/g_02500000"

def get_hifigan():
    conf = os.path.join("hifi-gan", "config_v1.json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan_model = "hifimodel_config_v1"
    os.system(f"wget '{hifigan_url}' -O {hifigan_model}")
    hifigan = Generator(h).to(torch.device("cpu"))
    state_dict_g = torch.load(hifigan_model, map_location=torch.device("cpu"))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser

def get_tacotron2():
    tacotron2_model = "MLPTTS"
    gdown.download(f"https://drive.google.com/uc?id={tacotron_id}", tacotron2_model, quiet=False)
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.max_decoder_steps = 3000
    hparams.gate_threshold = 0.25
    model = Tacotron2(hparams)
    state_dict = torch.load(tacotron2_model, map_location=torch.device("cpu"))['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model, hparams

# Load models
hifigan, h, denoiser = get_hifigan()
model, hparams = get_tacotron2()

def generate_speech(text):
    model.decoder.max_decoder_steps = 3000
    model.decoder.gate_threshold = 0.5
    with torch.no_grad():
        sequence = np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
        mel_outputs, mel_outputs_postnet, _, _ = model.inference(sequence)
        y_g_hat = hifigan(mel_outputs_postnet.float())
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
        audio_denoised = audio_denoised.cpu().numpy().reshape(-1)
        normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
        audio_denoised = audio_denoised * normalize
        speed_factor = 1.4
        audio_sped_up = resampy.resample(audio_denoised, hparams.sampling_rate,
                                         int(hparams.sampling_rate * speed_factor))
    return (int(hparams.sampling_rate * speed_factor), audio_sped_up.astype(np.int16))

# Gradio Interface
iface = gr.Interface(
    fn=generate_speech,
    inputs=gr.Textbox(label="Enter Text"),
    outputs=gr.Audio(label="Generated Speech"),
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
