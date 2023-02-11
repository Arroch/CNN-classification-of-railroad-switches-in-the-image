import cv2
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

def img_data_gen():

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        channel_shift_range=30.0,
        fill_mode="nearest",

    )

    for roots, dirs, files in os.walk("./ResNet_train"):
        i = 0

        if files != []:
            copys = 600 - len(files)
            repit = int((600 - len(files)) / len(files) + 1)
            print(roots)
            print(copys)
            print(repit)
            for file in files:
                image = os.path.join(roots[2:], file)
                img = cv2.imread(image)
                #img = cv2.resize(img, (100, 150))
                #cv2.imwrite('ResNet_train/img.png', img)
                img = img.reshape((1,) + img.shape)
                if i == copys:
                    break
                for batch in train_datagen.flow(img, batch_size=1):
                    cv2.imwrite(image + str(i) + '.png', (batch[0]))
                    i += 1
                    if i % repit == 0:
                        break


if __name__ == '__main__':
    img_data_gen()
