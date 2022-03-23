import os
import glob
import cv2
import time
import numpy as np
import json

import tensorflow as tf
from tensorflow.keras.losses import cosine_similarity, MSE

from image_utils import resizeAndPad

from transformers import ViTFeatureExtractor, ViTModel

TF_MODEL = "ResNet50"
IS_FAISS = False
IS_TFLITE = False
# IS_TRANSFORMER = True

if TF_MODEL == "ResNet50":
    from tensorflow.keras.applications.resnet50 import ResNet50 as EmbModel
    from tensorflow.keras.applications.resnet50 import preprocess_input
    EMB_SHAPE = 2048

elif TF_MODEL == "MobileNetV2":
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as EmbModel
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    EMB_SHAPE = 1280

if IS_FAISS:
    import faiss
else:
    import pynndescent


N_TAKE_IMGS = 5
DIST = cosine_similarity  # MSE

class ImageMetadata:
    def __init__(self,
                 img_name_key,
                 embs,
                 wikiart_url,
                 similarity=0):

        self.img_name_key = img_name_key

        self.embs = None
        if embs is not None:
            # self.embs = tf.keras.layers.GlobalAveragePooling2D()(embs)  # remove spatial

            # self.value = np.expand_dims(self.value, axis=2)  # needed for 1d pool
            # self.value = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(self.value)  # shrink a bit more

            # self.embs = np.squeeze(self.embs, axis=0)  # remove batch dim



            # transformer
            self.embs = embs.flatten()

        self.wikiart_url = wikiart_url.replace("\n", "")
        self.similarity = similarity
        self.source_img = None  # only needed for source img item


class Search():
    def __init__(self,
                 save_results_path=None, # where's results will be stored
                 embs_path=None,  # path to generated embeddings
                 is_use_graph=False, #  use graph or calculate distance yourself
                 is_server=False, #  send data to flask app
                 faiss_data_path=None):  # in other words if is server is False it means is test = True

        self.is_server = is_server
        self.img_metadata_list = None

        self.t_model = EmbModel(weights="imagenet", include_top=False)  #SearchModel()

        # transformer
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # end of transformer

        if IS_TFLITE is True:
            self.tf_av_dist = 0
            TFLITE_MODEL = "Mobile/saved_model/resnet50_float16.tflite"
            self.tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
            self.tflite_interpreter.allocate_tensors()

            self.input_details = self.tflite_interpreter.get_input_details()
            self.output_details = self.tflite_interpreter.get_output_details()

        self.embs_path = embs_path
        self.save_results_path = save_results_path

        self.img_metadata_list = []

        self.faiss_data_path = faiss_data_path

        if self.faiss_data_path is None:
            if ".npy" not in self.embs_path:
                npy_paths = glob.glob(os.path.join(self.embs_path, "*"))

                for npy_path in npy_paths:
                    self.img_metadata_list += self.open_test_npy(npy_path)

            else:
                self.img_metadata_list = self.open_test_npy(self.embs_path)
        else:
            with open(os.path.join(self.faiss_data_path, "wikiart_urls.txt")) as f:
                for line in f.readlines():
                    self.img_metadata_list.append(ImageMetadata(img_name_key=None,
                                                                embs=None,
                                                                wikiart_url=line))

        self.set_use_graph(is_use_graph)


    def get_source_img_data(self, source_img):
        source_img = resizeAndPad(source_img, (224, 224))
        # s_image = preprocess_input(source_img)
        # x = np.asarray(s_image).astype('float32')
        #
        # source_img_emb = self.t_model.predict(np.array([x]))  # np.array([]) for batch dimension

        #transformer
        inputs = self.feature_extractor(images=source_img, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        source_img_emb = last_hidden_states.detach().numpy().flatten()


        if IS_TFLITE is True:
            self.tflite_interpreter.set_tensor(self.input_details[0]['index'], np.array([x]))

            self.tflite_interpreter.invoke()

            tf_l_embs = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])

            dist = tf.math.reduce_mean(DIST(tf_l_embs, source_img_emb))
            if self.tf_av_dist != 0:
                self.tf_av_dist += dist
                self.tf_av_dist /= 2
            else:
                self.tf_av_dist += dist

            self.tflite_interpreter.reset_all_variables()

            source_img_emb = tf_l_embs  # temporary

        def_value = dict({"img_name_key": "Source",
                          "embs": source_img_emb,
                          "wikiart_url": "Source"})

        source_img_data = ImageMetadata(**def_value)

        source_img_data.source_img = source_img

        return source_img_data

    def calculate_embedding_distance(self, img_metadata_list, source_img_metadata):
        for index, img_metadata in enumerate(img_metadata_list):
            dist = tf.math.reduce_mean(DIST(img_metadata.embs,
                                            source_img_metadata.embs))
            img_metadata_list[index].similarity = dist.numpy()

    def set_use_graph(self,
                      is_use_graph,
                      is_store_faiss=False): #  data needed for faiss storing

        if is_use_graph:
            star_time = time.time()

            values = np.array(list(map(lambda x: x.embs, self.img_metadata_list)))

            if IS_FAISS:
                self.index = faiss.index_factory(EMB_SHAPE, "Flat")

                if self.faiss_data_path is None:
                    self.index.add(values)
                else:
                    self.index = faiss.read_index(os.path.join(self.faiss_data_path, "temp.faiss"))
                    print(self.index.ntotal)

                if is_store_faiss:
                    faiss.write_index(self.index, "temp.faiss")

                    with open('wikiart_urls.txt', 'w') as f:
                        for item in self.img_metadata_list:
                            f.write("%s\n" % item.wikiart_url)

            else:
                self.index = pynndescent.NNDescent(values, metric="cosine")
                self.index.prepare()

            print("graph init time", time.time() - star_time)
        else:
            self.index = None

    def open_test_npy(self, npy_path):
        img_metadata_list = []
        result_dict = np.load(npy_path, allow_pickle=True).item()

        for img_name_key, value in result_dict.items():
            if "embs" in value:
                if "is_broken" not in value:
                    # for compat with json db and test set
                    value["is_broken"] = False
                    value["image_url"] = "temp_url"

                img_metadata = ImageMetadata(img_name_key=img_name_key,
                                             embs=value["embs"],
                                             wikiart_url=value["wikiart_url"])

                img_metadata_list.append(img_metadata)

        return img_metadata_list

    def search(self,
               image_path,
               experiment_save_path="",
               expected_item_id=None):

        expected_item = None

        if expected_item_id is not None:
            exp_filtered_list = list(filter(lambda x: x.wikiart_url == expected_item_id,  self.img_metadata_list))
            if len(exp_filtered_list) == 0:
                print("EXPECTED IMG NOT IN img_metadata_list", expected_item_id)
                return
            else:
                expected_item = exp_filtered_list[0]

        image = cv2.imread(image_path)

        source_img_metadata = self.get_source_img_data(image)
        source_img_metadata.wikiart_url = expected_item_id

        if self.index is not None:
            #  Perform search on graph
            reshaped_values = source_img_metadata.embs.reshape((1, EMB_SHAPE))

            if IS_FAISS:
                neighbors = self.index.search(reshaped_values,
                                              N_TAKE_IMGS)  # actual search

                results = list(map(lambda x: self.img_metadata_list[x],
                                   neighbors[1][0]))  # map indexes to objects
            else:  # pydescent
                neighbors = self.index.query(reshaped_values,
                                             k=N_TAKE_IMGS)

                results = list(map(lambda x: self.img_metadata_list[x],
                                   neighbors[0][0]))  # map indexes to objects
        else:
            #  Perform iterations (for now, needed for distance calculations)
            self.calculate_embedding_distance(self.img_metadata_list, source_img_metadata)
            results = sorted(self.img_metadata_list,
                             key=lambda x: x.similarity,
                             reverse=False)[0:N_TAKE_IMGS]

        results_list = list(map(lambda x: {"url": x.wikiart_url,
                                           "dist": str(x.similarity)},
                                results))
        result_dict = {}

        for item in results_list:
            name = item['url']
            result_dict[name] = item["dist"]

        result_dict["expected"] = source_img_metadata.wikiart_url

        if self.is_server:
            return result_dict, results_list
        else:
            self.store_results(result_dict,
                               experiment_save_path)

            if expected_item_id is not None:
                pic_names = list(map(lambda x: x.wikiart_url, results))

                if IS_FAISS is False:
                    exp_dist = tf.math.reduce_mean(DIST(expected_item.embs,
                                                        source_img_metadata.embs))



                if expected_item.wikiart_url in pic_names:
                    return True, exp_dist
                else:
                    return False, exp_dist

    def store_results(self,
                      result_dict,
                      experiment_save_path):

        with open(experiment_save_path + ".json", 'w') as fp:
            json.dump(result_dict, fp)


if __name__ == "__main__":
    search = Search()
    search.set_use_graph(True)
    search.search(r"C:\Users\m\Desktop\624c53874b4ae54b4a5e85731f7c7b0b--the-block-infinity.jpg",
                  name="test_2")

