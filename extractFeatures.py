import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
max_pad_len = 500


def extract_features(file_name, model_type):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_best')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        if model_type == 2:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        if file_name == "dataSet/ambulance/sound_108.wav":
            librosa.display.specshow(mfcc, x_axis='time')
            plt.colorbar()
            plt.title('MFCC')
            plt.tight_layout()
            plt.savefig('sound_108.png')

        if model_type == 1:
            mfcc = np.mean(mfcc.T, axis=0)

    except Exception as e:
        return None

    return mfcc
