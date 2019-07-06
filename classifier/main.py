from __future__ import print_function
from keras import backend as K
import h5py

from GetData import get_data
from conf_matrix import plot_confusion_matrix
from convert_image import convert_image_to_array
from clf_plot import save_plot
from ImgClf import classifier

print('Getting data...')

(
    img_train,
    train_labels,
    img_test,
    test_labels,
    img_val,
    val_labels,
    img_ids,
    image_shape,
) = get_data()

epochs = 100
input_shape = image_shape
classes = set(train_labels)
num_classes = len(classes)

x_train, y_train = [], []
x_test, y_test = [], []
x_val, y_val = [], []

# Convert images --> arr and get their labels
print('Converting images...')
for img, label in zip(img_train, train_labels):
    print(f'converting img: {img}')
    x_train.append(convert_image_to_array(img))
    y_train.append(label)

for img, label in zip(img_test, test_labels):
    print(f'converting img: {img}')
    x_test.append(convert_image_to_array(img))
    y_test.append(label)

for img, label in zip(img_val, val_labels):
    print(f'converting img: {img}')
    x_val.append(convert_image_to_array(img))
    y_val.append(label)

print('Building model...')
model = classifier(input_shape, num_classes, epochs)

print('Running model...')
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=epochs,
    validation_data=(x_val, y_val)
)

print("Saving Model...")
model.save("cnn_model.h5")

save_plot(history)

score = model.evaluate(x_test, y_test, verbose=0)
loss, acc = score
print(f"\nTest loss - {loss}\nTest accuracy - {acc}")

print("Creating Confusion Matrix...")
y_pred = model.predict_classes(x_test)

plot_confusion_matrix(y_test, y_pred, classes)
