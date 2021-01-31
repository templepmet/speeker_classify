import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

torchaudio.set_audio_backend("sox_io")

filename = "voice/voice0.mp3"

waveform, sample_rate = torchaudio.load(filename)
fig = plt.figure()
plt.plot(waveform.t().numpy())

fig.savefig("img.png")