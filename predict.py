import os

from sklearn.preprocessing import LabelEncoder

import numpy as np

from createModel import create_model, create_model_cnn
from extractFeatures import extract_features


def load_trained_model(weights_path, class_count, model_type):
    if model_type == 1:
        model = create_model(class_count)
    else:
        model = create_model_cnn(class_count, 40, 500, 1)

    model.load_weights(weights_path)
    return model


def print_prediction(file_name):
    main_path = "dataSet/"
    dataset_name = input("Please enter dataset name: \n")

    if dataset_name not in ["cats_dogs", "EmergencySound", "Mixed"]:
        print("Please enter a valid dataset name.")
        exit()

    model_type = int(input("Select your model architecture (1 => Simple Architecture, 2 => CNN): \n"))

    if model_type not in [1, 2]:
        print("Please enter a valid model architecture number.")
        exit()

    prediction_feature = extract_features(file_name, model_type)

    directory_path = os.listdir(os.fsencode(main_path + "/" + dataset_name))

    y = np.array([str(i)[2:-1] for i in directory_path])

    le = LabelEncoder()
    le.fit_transform(y)

    if model_type == 1:
        model_architecture = "simple"
        prediction_feature = prediction_feature.reshape(-1, 1).transpose()
    else:
        model_architecture = "cnn"
        prediction_feature = prediction_feature.reshape(1, 40, 500, 1)

    model = load_trained_model('output_model/' + model_architecture + "/" + dataset_name + '.weights.best.basic_mlp.hdf5', len(y), model_type)

    predicted_vector = model.predict_classes(prediction_feature)

    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is:", predicted_class[0], '\n')

    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))


test_file_name = input("Enter your test data name: \n")
print_prediction("test/" + test_file_name + ".wav")
