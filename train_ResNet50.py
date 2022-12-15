from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
import resnet


batch_size = 16
nb_classes = 6
img_channels = 3
log_dir = 'logs/ResNet18/0043'
train_dir = "ResNet_train2"
# Каталог с данными для проверки
val_dir = "ResNet_val"
# Каталог с данными для теста
test_dir = "ResNet_test"
# Размеры изображения
img_rows, img_cols = 128, 128
# Размерность тензора на основе изображения для входных данных в нейронную сеть
input_shape = (img_rows, img_cols, img_channels)
# Количество эпох
epochs = 15
# Размер мини-выборки
batch_size = 16
# Количество изображений для обучения
nb_train_samples = 3697 # 1113
# Количество изображений для проверки
nb_validation_samples = 134
nb_test_samples = 121

datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')
checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-acc{acc:.3f}-val_acc{val_acc:.3f}.h5',
        monitor='val_acc', save_weights_only=True, save_best_only=True, period=3)
lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=1, verbose=1)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1)
logging = TensorBoard(log_dir=log_dir)
model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[logging, lr_reducer, early_stopper])#,callbacks=[lr_reducer, early_stopper])

model.save('ResNet18_1.h5')
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))

#https://doc-3c-ao-drive-data-export.googleusercontent.com/download/27vts92nkigl7s1giqsc92q8oeocvmjk/1o988bipemko68mkdgpgaj6jjpp6v1dg/1620526500000/52c1b6de-01d2-4e27-ae98-5d8c52c05a5d/104653801471132602345/ADt3v-Og4vXTKY2knGrgMt--Du5EprJafA4lVWdTsZwYH-kYPZBGmxuOcOnaZTcyv59hX9K8-1oO4ZWMxNQyOahlCzRRwEWBI9UFhjDUpaWWSjnhD8DKTD7bIZ9GxXeGxVy_NcY5_ZgHGkTRHKWNJmSLalT77H_YBIZxkg6Pn8G1Ky3Z0r3i4leqeSQBm8-LDbb82oQL4PXT6H1t3TQ0lGEZat3s8WUnj6ZsoxR90reshn3MujzgnLo3X9FmZDeV0HNH02JGmNiap9pG8uqj4_gKre1yxqWOUOOYCAAOXahQz8y4tPGRWYB10J-bheDS-ULMA3im2l2R?authuser=0&nonce=9cugior1bgdt0&user=104653801471132602345&hash=1lrvqq1a7jfg61hb4smf0b5qc51o0c1h