import os
import cv2
import json
import numpy as np
from google.colab.patches import cv2_imshow
import argparse

from image_utils import resizeAndPad, get_image_from_wikiart_url

input_imgs_path = "input_images"
save_results_path = "results"


def display_results(exp_res_folder_name,
                    exp_number,
                    input_imgs_path=input_imgs_path,
                    save_results_path=save_results_path):
    json_path = os.path.join(save_results_path, exp_res_folder_name, str(exp_number) + ".json")

    with open(json_path) as f:
        result_dict = json.load(f)

    input_img_path = os.path.join(input_imgs_path, "img_{}.png".format(exp_number))
    inp_img = cv2.imread(input_img_path)
    inp_img = resizeAndPad(inp_img, (224, 224))

    imgs = [inp_img]
    for value in result_dict.items():
        v_key = value[0]
        if v_key == "expected":
            image = get_image_from_wikiart_url(result_dict[v_key])
        else:
            image = get_image_from_wikiart_url(v_key)

        image = resizeAndPad(image, (224, 224))
        imgs.append(image)

    cv2_imshow(np.hstack(imgs))
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_res_folder_name', type=str)
    parser.add_argument('--exp_number', type=str)
    parser.add_argument('--input_imgs_path', type=str, default=input_imgs_path)
    parser.add_argument('--save_results_path', type=str, default=save_results_path)

    args = parser.parse_args()

    display_results(args.exp_res_folder_name,
                    args.exp_number,
                    args.input_imgs_path,
                    args.save_results_path)