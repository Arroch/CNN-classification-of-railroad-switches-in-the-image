import os
import json


def prep_train_txt():
    file_object = open('train_non_vision.txt', 'a')
    wmax = 0
    wmin = 1000
    hmax = 0
    hmin = 1000
    wmax2 = 0
    wmin2 = 1000
    hmax2 = 0
    hmin2 = 1000
    smax = 0
    smin = 100000000
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
                                if classTitle["tags"][0]["value"] != "no_vision":
                                    if (xmax-xmin)*(ymax-ymin) > smax:
                                        smax = (xmax-xmin)*(ymax-ymin)
                                        print('smax=', smax)
                                        print('h=', ymax - ymin)
                                        print('w=', xmax - xmin)

                                    if (xmax-xmin)*(ymax-ymin) < smin:
                                        smin = (xmax-xmin)*(ymax-ymin)
                                        print('smin=', smin)
                                        print('h=', ymax - ymin)
                                        print('w=', xmax - xmin)

                                    if xmax - xmin > wmax:
                                        wmax = xmax - xmin
                                        hmax = ymax - ymin
                                    if ymax - ymin > hmax2:
                                        hmax2 = ymax - ymin
                                        wmax2 = xmax - xmin
                                    if xmax - xmin < wmin:
                                        wmin = xmax - xmin
                                        hmin = ymax - ymin
                                    if ymax - ymin < hmin2:
                                        hmin2 = ymax - ymin
                                        wmin2 = xmax - xmin
                                    #print('w=', xmax - xmin)
                                    #print('h=', ymax - ymin)
                                    text = text + ' ' + str(xmin) + ',' + str(ymin) + ','\
                                                      + str(xmax) + ',' + str(ymax) + ','\
                                                      + str(0)
                            except IndexError:
                                print(path)
                                continue
                if text != img_path:
                    file_object.write(text + "\n")
    file_object.close()
    print(hmax, wmax)
    print(hmin, wmin)
    print(hmax2, wmax2)
    print(hmin2, wmin2)


prep_train_txt()
