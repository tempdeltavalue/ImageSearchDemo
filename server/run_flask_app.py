from flask import Flask, flash, request, redirect, url_for, render_template, json
from flask_ngrok import run_with_ngrok
import os
from werkzeug.utils import secure_filename
from datetime import datetime

from search import Search

import argparse

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
run_with_ngrok(app)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


class Item:
    def __init__(self, vals):
        self.__dict__ = vals


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        date_time_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        filename = date_time_str + ".png"
        dir_path = os.path.dirname(os.path.realpath(__file__))

        save_path = os.path.join(dir_path, app.config['UPLOAD_FOLDER'])
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        img_path = os.path.join(save_path, filename)
        file.save(img_path)

        search_save_path = os.path.join(save_path, date_time_str)

        result_dict, results_list = search_i.search(img_path)

        with open(search_save_path + ".json", 'w') as fp:
            json.dump(result_dict, fp)  # you should use Flask's json !

        flash('Image successfully uploaded and displayed below')

        return render_template('index.html',
                               filename=filename,
                               results=[Item(i) for i in results_list])
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static',
                            filename='uploads/' + filename),
                    code=301)


@app.route('/post_rating', methods=["GET", "POST"])
def post_rating():
    print("post rating")
    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--embs_path', type=str, default=None)
    parser.add_argument('--faiss_data', type=str, default=None)

    parser.add_argument('--is_use_graph', type=bool)

    args = parser.parse_args()

    search_i = Search(embs_path=args.embs_path,
                      faiss_data_path=args.faiss_data,
                      is_use_graph=args.is_use_graph,
                      is_server=True)
    app.run()
