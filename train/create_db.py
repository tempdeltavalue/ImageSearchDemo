import numpy as np
import glob
import os
import cv2
import time
import json

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as EmbModel
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from image_utils import resizeAndPad, get_image_from_wikiart_url

from urllib.request import urlopen

# Transformer thing
from transformers import ViTFeatureExtractor, ViTModel

splitter = "/"  # "\\"

#model = EmbModel(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Transformer thing
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

class DBEntryFromJSON:
    def __init__(self, wikiart_dict):
        self.url = wikiart_dict["image"]["url"]
        self.wikiart_url = wikiart_dict["url"]
        self.key = wikiart_dict["title"] + "___" + wikiart_dict["author"]

        self.wikiart_dict = wikiart_dict
        self.is_broken = False

    def img(self):
        try:
            resp = urlopen(self.url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            img = cv2.imdecode(image, -1)

            if len(img.shape) == 2:  # if image contains only one channel convert it to 3 channel image (for compatability)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            if img.shape[2] == 4:  # if image contains 4 channel (rgb+alpha) remove last one(alpha)
                img = img[:, :, :3]

            # removed for transformers !
            # img = resizeAndPad(img, (224, 224))

            return img

        except Exception as e:
            print(e)
            self.is_broken = True
            return np.zeros((224, 224, 3))


class DBEntry:
    def __init__(self, url):
        files = glob.glob(os.path.join(url, "*"))

        self.url = list(filter(lambda x: ".png" in x, files))[0] # needed for storing
        metadata_path = list(filter(lambda x: ".txt" in x, files))[0]

        self.key = self.url.split(splitter)[-1].split(".")[0]

        with open(metadata_path) as f:
            for line in f:
                self.wikiart_url = line.rstrip()

    def img(self):
        img = cv2.imread(self.url)

        # removed for transformers !
        # img = resizeAndPad(img, (224, 224))

        return img


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def preprocess(image):
    image = preprocess_input(np.array(image))#
    return image


def run_inference(model,  # model which is used for inrerence
                  img_data_list):  # img data on which inference will be performed
    start_time = time.time()

    chunk_len = 20
    result_dict = {}
    len_img_d_l = len(img_data_list)

    for ch_index, chunk in enumerate(chunks(img_data_list, chunk_len)):
        # preproc_imgs = np.array(list(map(lambda x: preprocess(x.img()),
        #                                  chunk)))

        # preproc_imgs = np.array(preproc_imgs)

        # embs = model.predict(preproc_imgs)

        # transformer thing
        parsed_imgs = list(map(lambda x: x.img(), chunk))
        inputs = feature_extractor(images=parsed_imgs, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embs = last_hidden_states.detach().numpy()
        # end of transformer thing

        start_ind = ch_index*chunk_len
        end_ind = start_ind+chunk_len

        print("ch_index {} from {}".format(ch_index, len_img_d_l / chunk_len))

        for index, x in enumerate(img_data_list[start_ind:end_ind]):
            key = x.key
            result_dict[key] = {}

            result_dict[key]["embs"] = np.expand_dims(embs[index], axis=0)
            result_dict[key]["wikiart_url"] = x.wikiart_url
            result_dict[key]["img_endpoint"] = x.url.split(splitter)[-2]  # index of item in search data

    print("inference time", time.time() - start_time)

    return result_dict


def create_npy(img_search_data_path,
               save_path,
               weights_path=None,
               chunk_count=1000):# 1000 # cannot open all files simultaneously

    if weights_path is not None:
        checkpoint = tf.train.Checkpoint(model)

        checkpoint.restore(weights_path)

    variant_imgs_url_list = glob.glob(os.path.join(img_search_data_path, "*"))

    result_dict = {}
    for index, val_chunk_urls in enumerate(chunks(variant_imgs_url_list, chunk_count)):
        variant_img_data_list = list(map(lambda x: DBEntry(x), val_chunk_urls))

        item_result_dict = run_inference(model, variant_img_data_list.copy())
        result_dict = dict(list(result_dict.items()) + list(item_result_dict.items()))

    np.save(save_path, result_dict)


def create_json_db_entry(string_line):
    wikiart_dict = json.loads(string_line)
    db_entry = DBEntryFromJSON(wikiart_dict)
    return db_entry


def create_npy_from_json(json_path,
                         base_save_path):
    chunk_count = 6000

    with open(json_path) as file:
        objects = file.read().splitlines()
        print("total len", len(objects))
        for index, val_chunk_urls in enumerate(chunks(objects, chunk_count)):
            print("chunk in", index)

            variant_img_data_list = list(map(lambda x: create_json_db_entry(x), val_chunk_urls))

            item_result_dict = run_inference(model, variant_img_data_list.copy())

            save_path = os.path.join(base_save_path, "json_db_{}.npy".format(index))

            np.save(save_path, item_result_dict)

            print("chunk {} from {}".format(index, len(objects)/chunk_count))

if __name__ == "__main__":
    img_dataset_path = r"C:\Users\m\Desktop\im_search_data" #r"C:\Users\m\Desktop\IMTestData\output_images"
    filename = "mobilenetv2.npy"

    base_save_path = r"C:\Users\m\Desktop\IMTestNpy"

    if os.path.exists(base_save_path) is False:
        os.mkdir(base_save_path)

    save_path = os.path.join(base_save_path, filename)

    create_npy(img_dataset_path, save_path)

