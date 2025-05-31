import matplotlib
matplotlib.use("Agg")
# python3 train_liveness.py --dataset dataset --model liveness.model --le le.pickle --plot plot.png
from model.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import tf2onnx
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Các tham số đầu vào
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='dataset',
                help="đường dẫn đến thư mục dataset")
ap.add_argument("-m", "--model", type=str, default='liveness.model',
                help="đường dẫn lưu mô hình ONNX")
ap.add_argument("-l", "--le", type=str, default='le.pickle',
                help="đường dẫn lưu label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="đường dẫn lưu biểu đồ loss/accuracy")
args = vars(ap.parse_args())

# Cài đặt các siêu tham số
INIT_LR = 1e-4
BS = 8
EPOCHS = 50

# Đọc dữ liệu khuôn mặt từ dataset
print("[INFO] đang tải ảnh...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (128, 128))
    data.append(image)
    labels.append(label)

# Chuẩn hóa dữ liệu ảnh về [0,1]
data = np.array(data, dtype="float") / 255.0

# Chuyển label thành one-hot
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

# Tính trọng số lớp để xử lý dữ liệu không cân bằng
class_weights = compute_class_weight('balanced', classes=np.unique(labels.argmax(axis=1)), y=labels.argmax(axis=1))
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Tăng cường augmentation dữ liệu
aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

# Chia tập train/test
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=42)

# Biên dịch mô hình
print("[INFO] đang biên dịch mô hình...")
opt = Adam(learning_rate=INIT_LR)
model = LivenessNet.build(width=128, height=128, depth=3, classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Cài đặt callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint('temp_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# Huấn luyện mô hình
print("[INFO] đang huấn luyện mô hình trong {} epoch...".format(EPOCHS))
H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // BS,
              epochs=EPOCHS,
              class_weight=class_weights_dict,
              callbacks=[reduce_lr, early_stopping, checkpoint],
              verbose=1)

# Đánh giá mô hình
print("[INFO] đang đánh giá mô hình...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=le.classes_))

# Chuyển mô hình sang ONNX
model_path = args["model"]
if not model_path.endswith('.onnx'):
    model_path += '.onnx'

print("[INFO] đang chuyển mô hình sang ONNX và lưu vào '{}'...".format(model_path))
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=None, opset=13)
onnx.save(onnx_model, model_path)

# Quantization mô hình ONNX
quantized_model_path = model_path.replace('.onnx', '_quantized.onnx')
print("[INFO] đang quantize mô hình và lưu vào '{}'...".format(quantized_model_path))
quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QUInt8)

# Lưu label encoder
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

# Vẽ biểu đồ loss/accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(H.history["loss"])), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, len(H.history["loss"])), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, len(H.history["loss"])), H.history["val_accuracy"], label="val_acc")
plt.title("Loss và Accuracy khi Huấn luyện")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# Xóa file mô hình tạm
if os.path.exists('temp_model.h5'):
    os.remove('temp_model.h5')