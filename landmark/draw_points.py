import cv2
import os
from config import *

def draw_points(lands, img, img_name, save_dir):
    for i in range(0, len(lands), 2):
        point = (lands[i], lands[i+1])
        cv2.circle(img, point, 1, (255, 0, 0))
    result_name = "result_"+img_name
    print(result_name)
    save_path = os.path.join(save_dir, result_name)
    print(save_path)
    cv2.imwrite(save_path, img)
    print('done')


if __name__ == '__main__':

    read_from_file = False
    img_path = './test_images/webwxgetmsgimg.png'
    image_txt = "./image_names.txt"
    image_dir = "./data/images/"
    resize_save_dir = "./results/"

    if read_from_file == True:
        with open(image_txt, 'r') as fin:
            for line in fin:
                line = line.strip.split(" ")
                image_name = line[0]
                land_marks = line[-1]
                img_path = os.path.join(image_dir, image_name)
                img = cv2.imread(img_path)
                img_resize = cv2.resize(img, args.height, args.width)
                draw_points(land_marks, img_resize, image_name)

    else:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
        #lands = [47, 30, 65, 31, 58, 38, 46, 43, 64, 43]
        #location :(140.268341,202.364410,161.242096,261.564728),
        #[Point(144, 100), Point(173, 100), Point(158, 112), Point(144, 119), Point(173, 117)]
        #Point(113, 151), Point(131, 151), Point(124, 169), Point(115, 179), Point(131, 179)]

        lands = [113,151,131,151,124,169,115,179,131,179]

        lands = [int(i) for i in lands]
        print('lands', lands)
        image_name = img_path.split("/")[-1]
        print(type(image_name))
        draw_points(lands, img, image_name, resize_save_dir)

