import math

from Config import config
from model import LayoutNet
from preprocessing import (
    VisualFeatureExtract,
    TextFeatureExtract,
    AttributeFeatureHandler,
)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

layoutnet = LayoutNet(config)


class LayoutNetDemo:
    def __init__(self, checkpoint_path):
        # define model
        self.layoutnet = LayoutNet(config)

        # restore from latest checkpoint
        self.layoutnet.load_weights(checkpoint_path).expect_partial()
        # checkpoint = tf.train.Checkpoint(model=self.layoutnet)
        # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

        # Save the model in SavedModel format with the serving function

        tf.saved_model.save(self.layoutnet, './saved_model')

        # visual feature extract
        self.vis_fea = VisualFeatureExtract()

        # text feature extract
        self.txt_fea = TextFeatureExtract()

        # category, text_ratio and image ratio handler
        self.sem_vec = AttributeFeatureHandler(keywords_path="./dataset/keywords.json")

    def generate(self, category, text_ratio, image_ratio, image_path, keywords_list, z):
        # process user input
        y, tr, ir = self.sem_vec.get(
            category=category, text_ratio=text_ratio, image_ratio=image_ratio
        )

        # extract image feature
        img_feature = self.vis_fea.extract(image_path)

        # extract text feature according to keywords
        # txt_feature = self.txt_fea.extract(keywords_list)
        txt_feature = self.txt_fea.extract(keywords_list)

        # generate result
        generated = self.layoutnet.generate(y, tr, ir, img_feature, txt_feature, z)
        generated = (generated + 1.0) / 2.0
        image = generated[0]

        return image


def demo():
    demo = LayoutNetDemo(checkpoint_path="./checkpoints/ckpt-300")

    # demo.layoutnet.save("layoutnet")
    category = "food"
    text_ratio = 0.5
    image_ratio = 0.5
    image_path1 = ["./demo/food.jpg", "./demo/wine.jpg"]
    keywords_list = ["Taste", "wine", "restaurant", "fruit", "market"]

    number_of_results = 9

    canva_row = round(math.sqrt(number_of_results))
    canva_col = math.ceil(float(number_of_results) / canva_row)
    canva = np.zeros((64 * canva_row, 64 * canva_col, 3), dtype=np.uint8)

    for i in range(number_of_results):
        row_idx = int(i / canva_col)
        col_idx = int(i % canva_col)
        z = np.random.normal(0.0, 1.0, size=(1, config.z_dim)).astype(np.float32)

        image_raw = demo.generate(
            category, text_ratio, image_ratio, image_path1, keywords_list, z
        )

        canva[
            row_idx * 64 : row_idx * 64 + 64, col_idx * 64 : col_idx * 64 + 64, :
        ] = np.uint8(image_raw * 255)


    image = Image.fromarray(canva)
    image.save("demo.png")

    plt.figure()
    plt.imshow(canva)


if __name__ == "__main__":
    demo()
