import os
import json
import random

import cv2

#from img_data_gen import img_data_gen


def prep_train():
    str_open = 0
    str_closed = 0
    str_right = 0
    str_left = 0
    str_forward = 0
    str_no_vision = 0
    path_err = []
    if not os.path.isdir('ResNet_train/open'):
        os.makedirs('ResNet_train/open')
    if not os.path.isdir('ResNet_train/closed'):
        os.makedirs('ResNet_train/closed')
    if not os.path.isdir('ResNet_train/right'):
        os.makedirs('ResNet_train/right')
    if not os.path.isdir('ResNet_train/left'):
        os.makedirs('ResNet_train/left')
    if not os.path.isdir('ResNet_train/forward'):
        os.makedirs('ResNet_train/forward')
    if not os.path.isdir('ResNet_train/no_vision'):
        os.makedirs('ResNet_train/no_vision')


    for roots, dirs, files in os.walk("./sort_img"):

        if roots[-3:] == "ann":
            for file in files:
                path = os.path.join(roots[2:], file)
                img_path = os.path.join(roots[2:-3], "img", file[: -5])

                with open(path, "r") as read_file:
                    data = json.load(read_file)
                    for classTitle in data["objects"]:
                        if classTitle["classTitle"] == "PoStrelka" or classTitle["classTitle"] == "ProtivStrelka":
                            xmin,ymin = classTitle["points"]["exterior"][0]
                            xmax,ymax = classTitle["points"]["exterior"][1]
                            image_path = cv2.imread(img_path)
                            if (xmax-xmin)*(ymax-ymin) >= 2048*1080*0.01:
                                try:
                                    if classTitle["tags"][0]["value"] == "open":
                                        #cv2.imwrite('ResNet_train/open/open.' + str(str_open) + '.png', image_path[ymin:ymax, xmin:xmax])
                                        str_open += 1
                                    if classTitle["tags"][0]["value"] == "closed":
                                        #cv2.imwrite('ResNet_train/closed/closed.' + str(str_closed) + '.png', image_path[ymin:ymax, xmin:xmax])
                                        cv2.imwrite('ResNet_train/1closed.' + str(str_closed) + '.png',
                                                    image_path)
                                        str_closed += 1
                                    if classTitle["tags"][0]["value"] == "right":
                                        #cv2.imwrite('ResNet_train/right/right.' + str(str_right) + '.png', image_path[ymin:ymax, xmin:xmax])
                                        cv2.imwrite('ResNet_train/1right.' + str(str_right) + '.png',
                                                    image_path)
                                        str_right += 1
                                    if classTitle["tags"][0]["value"] == "left":
                                        #cv2.imwrite('ResNet_train/left/left.' + str(str_left) + '.png', image_path[ymin:ymax, xmin:xmax])
                                        cv2.imwrite('ResNet_train/1left.' + str(str_left) + '.png',
                                                    image_path)
                                        str_left += 1
                                    if classTitle["tags"][0]["value"] == "forward":
                                        #cv2.imwrite('ResNet_train/forward/forward.' + str(str_forward) + '.png', image_path[ymin:ymax, xmin:xmax])
                                        str_forward += 1
                                    if classTitle["tags"][0]["value"] == "no_vision" and (xmax-xmin)*(ymax-ymin) >= 2048*1080*0.01:
                                        #cv2.imwrite('ResNet_train/no_vision/no_vision.' + str(str_no_vision) + '.png', image_path[ymin:ymax, xmin:xmax])
                                        str_no_vision += 1

                                except IndexError:
                                    print(path)
                                    continue

    print('open=', str_open)
    print('closed=', str_closed)
    print('right=', str_right)
    print('left=', str_left)
    print('forward=', str_forward)
    print('no_vision=', str_no_vision)
    print(path_err)

def prep_val():
    #l = list(range(0, 2999))
    #random.shuffle(l)
    if not os.path.isdir('ResNet_val/open'):
        os.makedirs('ResNet_val/open')
    if not os.path.isdir('ResNet_val/closed'):
        os.makedirs('ResNet_val/closed')
    if not os.path.isdir('ResNet_val/right'):
        os.makedirs('ResNet_val/right')
    if not os.path.isdir('ResNet_val/left'):
        os.makedirs('ResNet_val/left')
    if not os.path.isdir('ResNet_val/forward'):
        os.makedirs('ResNet_val/forward')
    if not os.path.isdir('ResNet_val/no_vision'):
        os.makedirs('ResNet_val/no_vision')
    for roots, dirs, files in os.walk("./ResNet_train"):
        #print(roots, dirs, files)
        if files != []:
            print(int(len(files) * 0.1))
            val_data = int(len(files) * 0.1)
            for i in range(int(val_data)):
                path = os.path.join(roots[2:], files[i])
                os.replace(path, os.path.join("ResNet_val", path[13:]))
def prep_test():
    #l = list(range(0, 2700))
    #random.shuffle(l)
    if not os.path.isdir('ResNet_test/open'):
        os.makedirs('ResNet_test/open')
    if not os.path.isdir('ResNet_test/closed'):
        os.makedirs('ResNet_test/closed')
    if not os.path.isdir('ResNet_test/right'):
        os.makedirs('ResNet_test/right')
    if not os.path.isdir('ResNet_test/left'):
        os.makedirs('ResNet_test/left')
    if not os.path.isdir('ResNet_test/forward'):
        os.makedirs('ResNet_test/forward')
    if not os.path.isdir('ResNet_test/no_vision'):
        os.makedirs('ResNet_test/no_vision')
    for roots, dirs, files in os.walk("./ResNet_train"):
        #print(roots, dirs, files)
        if files != []:
            print(int(len(files) * 0.1))
            val_data = int(len(files) * 0.1)
            for i in range(int(val_data)):
                path = os.path.join(roots[2:], files[i])
                os.replace(path, os.path.join("ResNet_test", path[13:]))
prep_train()
#img_data_gen()
#random.seed(7)
#prep_val()
#prep_test()