import glob
import os

import numpy as np
import pandas as pd

from extractFeatures import extract_features

from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(directory, model_type):
    features = []

    directory_path = os.fsencode(directory)

    for class_folder in os.listdir(directory_path):
        class_label = str(class_folder)[2:-1]
        wav_files = glob.glob(directory + "/" + class_label + "/*.wav", recursive=True)

        for sound_file in wav_files:
            file_name = os.path.join(sound_file)
            data = extract_features(file_name, model_type)

            if data is not None:
                features.append([data, class_label])

    features_data = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(features_data), ' files')

    X = np.array(features_data.feature.tolist())
    y = np.array(features_data.class_label.tolist())

    le = LabelEncoder()
    yy = to_categorical(le.fit_transform(y))

    result_data = [train_test_split(X, yy, test_size=0.2, random_state=42), yy.shape[1]]

    return result_data


