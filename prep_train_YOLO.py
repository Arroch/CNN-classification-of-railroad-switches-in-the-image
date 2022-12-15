import os
import json


def prep_train_txt():
    file_object = open('train_YOLO.txt', 'a')
    classes = {
        "closed": 0,
        "forward": 1,
        "left": 2,
        "no_vision": 3,
        "open": 4,
        "right": 5,

    }

    for roots, dirs, files in os.walk("./sort_img"):

        if roots[-3:] == "ann":
            for file in files:
                path = os.path.join(roots[2:], file)
                img_path = os.path.join(roots[2:-3], "img", file[: -5])
                text = img_path

                with open(path, "r") as read_file:
                    data = json.load(read_file)
                    for classTitle in data["objects"]:
                        if classTitle["classTitle"] == "PoStrelka" or classTitle["classTitle"] == "ProtivStrelka":
                            xmin,ymin = classTitle["points"]["exterior"][0]
                            xmax,ymax = classTitle["points"]["exterior"][1]

                            #if (xmax-xmin)*(ymax-ymin) >= 2048*1084*0.01:
                            try:

                                text = text + ' ' + str(xmin) + ',' + str(ymin) + ','\
                                                  + str(xmax) + ',' + str(ymax) + ','\
                                                  + str(classes[classTitle["tags"][0]["value"]])
                            except IndexError:
                                print(path)
                                continue
                if text != img_path:
                    file_object.write(text + "\n")
    file_object.close()

prep_train_txt()
