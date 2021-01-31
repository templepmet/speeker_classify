import librosa

# Audio Data
audio_path = 'voice/voice0.wav'

# Load
data, sr = librosa.load(audio_path,
                        sr=16000)

# MFCC(メル周波数ケプストラム係数)に変換
mfcc = librosa.feature.mfcc(data,
                            sr=sr,
                            n_mfcc=40,
                            n_mels=128,
                            win_length=480,
                            hop_length=160,
                            n_fft=512,
                            dct_type=2)

print(mfcc.shape)  # --> (40, 101)

# plot
import librosa.display
import matplotlib.pyplot as plt

librosa.display.specshow(mfcc,
                         sr=sr,
                         hop_length=160,
                         x_axis='time')

plt.colorbar()
plt.title('voice')
plt.tight_layout()
plt.show()