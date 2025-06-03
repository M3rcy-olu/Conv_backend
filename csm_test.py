import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "csm"))
from generator import load_csm_1b, Segment
import torchaudio
import torch
import pyaudio
import numpy as np
from typing import List

from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

# Save the audio to audio.wav file
# torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

audio_np = audio.cpu().numpy()
audio_int16 = (audio_np * 32767).astype(np.int16)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=generator.sample_rate,
                output=True)
                
stream.write(audio_int16.tobytes())
stream.stop_stream()
stream.close()
p.terminate()
