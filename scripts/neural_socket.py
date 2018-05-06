# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import glob
import os

from flask import Flask
from flask import render_template
from flask import send_from_directory
from flask_restful import Resource, Api
app = Flask(__name__)
api = Api(app)

model_file = r"C:\Users\SP\Documents\WORK\GP\tensorflow\tensorflow-for-poets-2\tf_files\retrained_graph.pb"
label_file = r"C:\Users\SP\Documents\WORK\GP\tensorflow\tensorflow-for-poets-2\tf_files\retrained_labels.txt"
screenshot_dir = r"C:\Users\SP\Documents\WORK\GP\castle_crusade_prototype\Screenshots"

scores_team1 = [0, 0, 0, 0, 0]
scores_team2 = [0, 0, 0, 0, 0]


class Inference(Resource):
    def get(self, team=1):
        def read_tensor_from_image_file(file_path, input_height=224, input_width=224, input_mean=128, input_std=128):
            file_reader = tf.read_file(file_path, "file_reader")
            if file_path.endswith(".png"):
                image_reader = tf.image.decode_png(file_reader, channels=3, name='png_reader')
            elif file_path.endswith(".gif"):
                image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
            elif file_path.endswith(".bmp"):
                image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
            else:
                image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')

            float_caster = tf.cast(image_reader, tf.float32)
            dims_expander = tf.expand_dims(float_caster, 0)
            resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
            normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
            return tf.Session().run(normalized)

        def load_labels(label_file):
            label = []
            proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
            for l in proto_as_ascii_lines:
                label.append(l.rstrip())
            return label

        print("inference called, team: " + str(team))

        output_values = []
        for file_path in glob.glob(screenshot_dir + '\Team' + str(team) + "\*.jpg"):
            t = read_tensor_from_image_file(file_path)

            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]
            labels = load_labels(label_file)

            total = 0
            for i in top_k:
                total += int(labels[i]) * results[i]

            output_values.append(format(total, '.3f'))

        if team == 1:
            global scores_team1
            scores_team1 = output_values
        elif team == 2:
            global scores_team2
            scores_team2 = output_values

        return ','.join(map(str, output_values))


class Image(Resource):
    def get(self, team, filename):
        path = screenshot_dir + '\Team' + str(team)
        return send_from_directory(path, filename)


api.add_resource(Inference, '/inference/<int:team>')
api.add_resource(Image, '/image/<int:team>/<path:filename>')


@app.route("/<int:team>", methods=['GET'])
def index(team=1):
    def calculate_active(team_scores):
        sorted = team_scores[:]
        sorted.sort(reverse=True)
        threshold = sorted[1]

        active = []
        for i, v in enumerate(team_scores):
            active.append(v >= threshold)
        return active

    if team == 1:
        scores = scores_team1
    elif team == 2:
        scores = scores_team2

    ai_active = calculate_active(scores)
    images = os.listdir(screenshot_dir + '\Team' + str(team))
    return render_template('index.html', team=team, images=images, scores=scores, ai_active=ai_active)


if __name__ == "__main__":
    def load_graph():
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    print("initialise")
    graph = load_graph()
    input_operation = graph.get_operation_by_name("import/input")
    output_operation = graph.get_operation_by_name("import/final_result")
    sess = tf.Session(graph=graph)
    print("done")

    # set-up website
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False, threaded=True)

