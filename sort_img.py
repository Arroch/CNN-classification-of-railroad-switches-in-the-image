import os
import json
import cv2


def sort_img():
    i =0
    for roots, dirs, files in os.walk("./img"):

        if roots[-3:] == "ann":
            for file in files:
                path = os.path.join(roots[2:], file)
                img_path = os.path.join(roots[2:-3], "img", file[: -5])
                with open(path, "r") as read_file:
                    data = json.load(read_file)
                    for classTitle in data["objects"]:
                        if os.path.isfile(os.path.join("large_im", img_path[4:])):
                            continue
                        if classTitle["classTitle"] == "PoStrelka" or classTitle["classTitle"] == "ProtivStrelka":
                            xmin, ymin = classTitle["points"]["exterior"][0]
                            xmax, ymax = classTitle["points"]["exterior"][1]

                            if (xmax - xmin) * (ymax - ymin) >= 2048 * 1084 * 0.05:

                                i+=1
                                image_path = cv2.imread(img_path)
                                cv2.imwrite("large_im/img" + str(i) + '.png', image_path)
                                print("PATH", os.path.join(roots[2:], file))





sort_img()