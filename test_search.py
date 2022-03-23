import os
import glob
import argparse
# from concurrent.futures.thread import ThreadPoolExecutor

import time
from search import Search

def main(input_imgs_path,
         embs_path,
         search_data_path,
         save_results_path,
         name,
         faiss_data_path=None):

    search_i = Search(embs_path=embs_path,
                      save_results_path=save_results_path,
                      is_server=False,
                      is_use_graph=False,
                      faiss_data_path=faiss_data_path)

    paths = glob.glob(os.path.join(input_imgs_path, "*"))
    start_time = time.time()

    futures = []
    # with ThreadPoolExecutor(max_workers=2) as executor:
    for img_path in paths:
        print(img_path)
        splitter = "/"
        index = img_path.split(splitter)[-1].split("_")[-1].split(".")[0]

        files = glob.glob(os.path.join(search_data_path, index, "*"))
        txt_file = list(filter(lambda x: ".txt" in x, files))[0]
        with open(txt_file) as f:
            expected_item_id = f.readlines()[0].replace("\n", "")

        save_path = os.path.join(save_results_path, name)

        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        save_path = os.path.join(save_path, index)

        # future = executor.submit(search_i.search, img_path, save_path, expected_item_id)
        future = search_i.search(image_path=img_path, # pseudo future
                                 experiment_save_path=save_path,
                                 expected_item_id=None)

        futures.append(future)

        # executor.shutdown(wait=True)

        contains_expected = 0
        #
        # for f in futures:
        #     is_top5_correct, exp_dist = f#.result()
        #     if is_top5_correct is True:
        #         contains_expected += 1

    print("exec time", time.time() - start_time)
    print("Acc", contains_expected / len(search_i.img_metadata_list))
    # print("total sum_distance_to_expected", search_i.sum_distance_to_expected / len(search_i.img_metadata_list))
    # print("total sum_distance_to_opposite", search_i.total_distance / len(search_i.img_metadata_list))
    # print("total tf_av", search_i.tf_av_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_imgs_path', type=str)
    parser.add_argument('--embs_path', type=str)
    parser.add_argument('--faiss_data', type=str, default=None)
    parser.add_argument('--search_data_path', type=str)
    parser.add_argument('--save_results_path', type=str)
    parser.add_argument('--name', type=str, default="temp")

    args = parser.parse_args()

    main(input_imgs_path=args.input_imgs_path,
         embs_path=args.embs_path,
         search_data_path=args.search_data_path,
         save_results_path=args.save_results_path,
         name=args.name,
         faiss_data_path=args.faiss_data)
