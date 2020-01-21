from createModel import create_model, create_model_cnn
from loadData import load_data

from keras.callbacks import ModelCheckpoint
from datetime import datetime

main_path = "dataSet/"

dataset_name = input("Please enter dataset name: \n")

if dataset_name not in ["cats_dogs", "EmergencySound", "Mixed"]:
    print("Please enter a valid dataset name.")
    exit()

model_type = int(input("Select your model architecture (1 => Simple Architecture, 2 => CNN): \n"))

if model_type not in [1, 2]:
    print("Please enter a valid model architecture number.")
    exit()

data_list, num_labels = load_data(main_path + dataset_name, model_type)

x_train, x_test, y_train, y_test = data_list

filter_size = 2

if model_type == 1:
    model_architecture = "simple"
    model = create_model(num_labels)
else:
    model_architecture = "cnn"
    model = create_model_cnn(num_labels, 40, 500, 1)

    x_train = x_train.reshape(x_train.shape[0], 40, 500, 1)
    x_test = x_test.reshape(x_test.shape[0], 40, 500, 1)

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Accuracy before training: %.4f%%" % accuracy)

num_epochs = 100
num_batch_size = 32

check_pointer = ModelCheckpoint(filepath='output_model/' + model_architecture + "/" + dataset_name + '.weights.best.basic_mlp.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
          callbacks=[check_pointer], verbose=1)


duration = datetime.now() - start
print("Total Training time: ", duration)

score = model.evaluate(x_train, y_train, verbose=0)
print("Accuracy Training Set: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy Testing Set: ", score[1])
