import os

os.environ['GENSIM_DATA_DIR'] = './'
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import gensim.downloader as api
from gensim.models import Word2Vec

#model = Word2Vec.load('word2vec-google-news-300')

class VisualFeatureExtract:
    def __init__(self):
        self.vgg16_base = tf.keras.applications.VGG16(include_top=False,
                                                      input_shape=(224, 224,
                                                                   3), weights=None)
        weights_path = './vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.vgg16_base.load_weights(weights_path)
        self.vgg16_feature_extract = tf.keras.Model(
            inputs=self.vgg16_base.input,
            outputs=self.vgg16_base.get_layer('block5_conv3').output)

    def extract(self, img_list, target_size=(224, 224)):
        """Extract feature from image list using VGG16.

        Args:
            img_list (list of string of Image): The image list of the magazine page. It should be string (for image path) or Image (if you have read images by PIL).
            target_size (tuple, optional): The target input size of the model. Defaults to (224, 224).

        Returns:
            numpy array: The feature of the input image list. Size is (14, 14, 512).
        """
        # if img_list is a list of string
        # means it contains image path
        new_img_list = []
        if type(img_list[0]) is str:
            for item in img_list:
                img_item = image.load_img(item, target_size=target_size)
                new_img_list.append(img_item)
        # or item in img_list should be PIL Image
        else:
            try:
                for item in img_list:
                    new_img_list.append(item.resize(target_size))
            except Exception as e:
                print('Visual Feature Extract Error: %s' % e)

        for idx, item in enumerate(new_img_list):
            new_img_list[idx] = image.img_to_array(item)

        # pack the images list into a "batch"
        new_img_list = np.stack(new_img_list)

        # it's important to preprocess
        new_img_list = tf.keras.applications.vgg16.preprocess_input(
            new_img_list)

        result = self.vgg16_feature_extract(new_img_list)
        result = np.array(result).sum(axis=0)
        result = np.stack([result])

        return result


class TextFeatureExtract:
    def __init__(self):
        pass
        # get more infomation about the model on
        # https://github.com/RaRe-Technologies/gensim-data
        # self.model = api.load('word2vec-google-news-300')

    def extract(self, word_list):
        '''
        result = []
        for item in word_list:
            result.append(np.array(item))
        result = np.array(result)
        result = result.sum(axis=0)
        result = np.expand_dims(result, axis=0)

        return result
        '''
        result = []
        for word in word_list:
            # Generate a one-hot encoding for each word
            one_hot = np.zeros(100)  # Assuming 100-dimensional feature vector
            # For demonstration, let's pretend each word corresponds to a specific index
            index = hash(word) % 100  # Map word to an index
            one_hot[index] = 1
            result.append(one_hot)

        result = np.array(result)
        # Summing up the one-hot encodings
        result = result.sum(axis=0)
        result = np.expand_dims(result, axis=0)
        return result

    def word_fea_extract(self, word):
        return np.zeros(100)


class AttributeFeatureHandler:
    # category:     int64 0-5
    # text ratio:   0.1 - 0.7  7 scale -> int 0-6
    # image ratio:  0.1 - 1.0 10 scale -> int 0-9
    def __init__(self, keywords_path='./dataset/keywords.json'):
        # read category list from keywords.json
        f = open(keywords_path)
        category_json = json.load(f)
        self.category_list = list(category_json.keys())

    def get(self, category, text_ratio, image_ratio):
        assert type(category) == str
        assert type(text_ratio) == float and text_ratio >= 0
        assert type(image_ratio) == float and image_ratio >= 0

        # embedding category using integer
        category = category.lower()
        if category not in self.category_list:
            assert ValueError('%s is not a valid category' % category)

        # get the index of the category as its embedding
        cate_embedding = self.category_list.index(category)

        tr_embedding = min(round(text_ratio * 10), 6)
        ir_embedding = min(round(image_ratio * 10), 9)

        return cate_embedding, tr_embedding, ir_embedding
