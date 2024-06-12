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
import os
import time

layoutnet = LayoutNet(config)

def sample(step):
    # get data from dataset
    resized_image, label, textRatio, imgRatio, visualfea, textualfea = \
        dataset.next()

    z = np.random.normal(0.0, 1.0, size=(config.batch_size,
                                         config.z_dim)).astype(np.float32)

    # get all results from the model
    z_mean, z_log_sigma_sq, E, G, G_recon, D_real, D_fake = layoutnet(
        resized_image,
        label,
        textRatio,
        imgRatio,
        visualfea,
        textualfea,
        z,
        is_training=False)

    # save the generated image
    G_recon = (G_recon + 1.0) / 2.0
    image = G_recon[0]
    image = np.uint8(image * 255)
    image = Image.fromarray(image)
    image.save(config.sampledir + "/sample_{}.png".format(step))

    # save the real image
    resized_image = (resized_image + 1.0) / 2.0
    image = resized_image[0]
    image = np.uint8(image * 255)
    image = Image.fromarray(image)
    image.save(config.sampledir + "/real_{}.png".format(step))

    # save the visual feature
    visualfea = visualfea[0]
    visualfea = np.uint8(visualfea)
    visualfea = Image.fromarray(visualfea)
    visualfea.save(config.sampledir + "/visualfea_{}.png".format(step))

    # save the textual feature
    textualfea = textualfea[0]
    textualfea = np.uint8(textualfea)
    textualfea = Image.fromarray(textualfea)
    textualfea.save(config.sampledir + "/textualfea_{}.png".format(step))

    # save the semantic vector
    semvec = label[0]
    semvec = np.uint8(semvec)
    semvec = Image.fromarray(semvec)
    semvec.save(config.sampledir + "/semvec_{}.png".format(step))

    print("Sample saved at step {}".format(step))


# decode function for dataset
def _decode_tfrecords(example_string):
    features = tf.io.parse_single_example(
        example_string,
        features={
            "label": tf.io.FixedLenFeature([], tf.int64),
            "textRatio": tf.io.FixedLenFeature([], tf.int64),
            "imgRatio": tf.io.FixedLenFeature([], tf.int64),
            'visualfea': tf.io.FixedLenFeature([], tf.string),
            'textualfea': tf.io.FixedLenFeature([], tf.string),
            "img_raw": tf.io.FixedLenFeature([], tf.string)
        })

    image = tf.io.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [60, 45, 3])
    image = tf.cast(image, tf.float32)

    resized_image = tf.image.resize_with_crop_or_pad(image, 64, 64)
    resized_image = resized_image / 127.5 - 1.

    label = tf.cast(features['label'], tf.int32)

    textRatio = tf.cast(features['textRatio'], tf.int32)
    imgRatio = tf.cast(features['imgRatio'], tf.int32)

    visualfea = tf.io.decode_raw(features['visualfea'], tf.float32)
    visualfea = tf.reshape(visualfea, [14, 14, 512])

    textualfea = tf.io.decode_raw(features['textualfea'], tf.float32)
    textualfea = tf.reshape(textualfea, [300])

    return resized_image, label, textRatio, imgRatio, visualfea, textualfea


dataset = tf.data.TFRecordDataset(config.filenamequeue)
dataset = dataset.map(_decode_tfrecords)

dataset = dataset.shuffle(buffer_size=4096,
                            reshuffle_each_iteration=True)
dataset = dataset.repeat()
dataset = dataset.batch(batch_size=config.batch_size)
dataset = dataset.as_numpy_iterator()

# Discriminator loss is same as the least square GAN (LSGAN)
def discriminator_loss(D_real, D_fake):
    loss_D_real = tf.reduce_mean(tf.nn.l2_loss(D_real - tf.ones_like(D_real)))
    loss_D_fake = tf.reduce_mean(tf.nn.l2_loss(D_fake - tf.zeros_like(D_fake)))

    loss_D = loss_D_real + loss_D_fake

    return loss_D

# Generator loss should have 3 parts:
# 1. loss_Gls - G loss in LSGAN
# 2. recon_loss - reconstruction loss
# 3. Variety Loss mentioned in the paper, but not implemented in author's code
def generator_loss(x, z_log_sigma_sq, z_mean, D_fake, G_recon):
    # loss_Gls is the loss function for G in LSGAN
    loss_Gls = tf.reduce_mean(tf.nn.l2_loss(D_fake - tf.ones_like(D_fake)))

    orig_input = tf.reshape(x, [config.batch_size, 64 * 64 * 3])
    orig_input = (orig_input + 1) / 2.
    generated_flat = tf.reshape(G_recon, [config.batch_size, 64 * 64 * 3])
    generated_flat = (generated_flat + 1) / 2.

    recon_loss = tf.reduce_sum(tf.pow(generated_flat - orig_input, 2), 1)

    recon_loss = tf.reduce_mean(recon_loss) / 64 / 64 / 3

    loss_G = loss_Gls + recon_loss

    return loss_G

# Encoder loss should have 2 parts:
# 1. kl_div - KL divergence loss
# 2. recon_loss - reconstruction loss
def encoder_loss(x, z_log_sigma_sq, z_mean, G_recon):
    kl_div = -0.5 * tf.reduce_sum(
        1 + 2 * z_log_sigma_sq - tf.square(z_mean) -
        tf.exp(2 * z_log_sigma_sq), 1)

    orig_input = tf.reshape(x, [config.batch_size, 64 * 64 * 3])
    orig_input = (orig_input + 1) / 2.
    generated_flat = tf.reshape(G_recon, [config.batch_size, 64 * 64 * 3])
    generated_flat = (generated_flat + 1) / 2.

    recon_loss = tf.reduce_sum(tf.pow(generated_flat - orig_input, 2), 1)

    loss_E = tf.reduce_mean(kl_div + recon_loss) / 64 / 64 / 3

    return loss_E

# define optimizer
# all optimizer use Adam

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr,
                                               beta_1=config.beta1)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr,
                                                   beta_1=config.beta1)
encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr,
                                             beta_1=config.beta1)

# define checkpoint
checkpoint_prefix = os.path.join(config.checkpoint_dir,
                                 config.checkpoint_basename)

checkpoint = tf.train.Checkpoint(
                                generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                encoder_optimizer=encoder_optimizer,
                                generator=layoutnet.generator,
                                discriminator=layoutnet.discriminator,
                                encoder=layoutnet.encoder)

# define checkpoint manager
manager = tf.train.CheckpointManager(checkpoint,
                                     config.checkpoint_dir,
                                     max_to_keep=3)

def train_step(z,
               is_training=True,
               discriminator=True,
               generator=True,
               encoder=True):
    # get data from dataset
    resized_image, label, textRatio, imgRatio, visualfea, textualfea = dataset.next()

    with tf.GradientTape() as disc_tape, tf.GradientTape(
    ) as gen_tape, tf.GradientTape() as encoder_tape:
        # get all results from the model
        z_mean, z_log_sigma_sq, E, G, G_recon, D_real, D_fake = layoutnet(
            resized_image,
            label,
            textRatio,
            imgRatio,
            visualfea,
            textualfea,
            z,
            is_training=is_training)

        # calculate loss
        disc_loss = discriminator_loss(D_real, D_fake)
        gen_loss = generator_loss(x=resized_image,
                                  z_log_sigma_sq=z_log_sigma_sq,
                                  z_mean=z_mean,
                                  D_fake=D_fake,
                                  G_recon=G_recon)
        encod_loss = encoder_loss(x=resized_image,
                                  z_log_sigma_sq=z_log_sigma_sq,
                                  z_mean=z_mean,
                                  G_recon=G_recon)

    if is_training:
        # get the trainable variables
        # BP algorithm will modify these variables
        discriminator_variables = layoutnet.discriminator.trainable_variables
        generator_variables = layoutnet.generator.trainable_variables
        # when training encoder, we not only modify encoder's weights
        # but also modify weights of embedding layers
        encoder_variables = layoutnet.encoder.trainable_variables + \
                            layoutnet.embeddingImg.trainable_variables + \
                            layoutnet.embeddingTxt.trainable_variables + \
                            layoutnet.embeddingSemvec.trainable_variables + \
                            layoutnet.embeddingFusion.trainable_variables

        # there will be 3 kinds of training
        # we use flags to control training process

        # 1. train disciminator
        if discriminator:
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, discriminator_variables)
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, discriminator_variables))

        # 2. train generator
        if generator:
            gradients_of_generator = gen_tape.gradient(gen_loss,
                                                       generator_variables)
            generator_optimizer.apply_gradients(
                zip(gradients_of_generator, generator_variables))

        # 3. train encdoer
        if encoder:
            gradients_of_encoder = encoder_tape.gradient(
                encod_loss, encoder_variables)
            encoder_optimizer.apply_gradients(
                zip(gradients_of_encoder, encoder_variables))

    return disc_loss, gen_loss, encod_loss

def train():
    for step in range(config.max_steps):
        t1 = time.time()

        z = np.random.normal(0.0, 1.0, size=(config.batch_size,
                                             config.z_dim)).astype(np.float32)

        # train disc
        disc_loss, gen_loss, encod_loss = train_step(z=z,
                                                     discriminator=True,
                                                     generator=False,
                                                     encoder=False)

        # train gen
        disc_loss, gen_loss, encod_loss = train_step(z=z,
                                                     discriminator=False,
                                                     generator=True,
                                                     encoder=False)

        # train encoder
        disc_loss, gen_loss, encod_loss = train_step(z=z,
                                                     discriminator=False,
                                                     generator=False,
                                                     encoder=True)

        # train gen and encoder
        # make sure discriminator cannot distinguish fake images
        while disc_loss < 1.:
            disc_loss, gen_loss, encod_loss = train_step(z,
                                                         discriminator=False,
                                                         generator=True,
                                                         encoder=True)

        t2 = time.time()

        if (step + 1) % config.summary_every_n_steps == 0:
            disc_loss, gen_loss, encod_loss = train_step(z, is_training=False)
            print("step {:5d},loss = (G: {:.8f}, D: {:.8f}), E: {:.8f}".format(
                step + 1, gen_loss, disc_loss, encod_loss))

        if (step + 1) % config.sample_every_n_steps == 0:
            eta = (t2 - t1) * (config.max_steps - step + 1)
            print("Finished {}/{} step, ETA:{:.2f}s".format(
                step + 1, config.max_steps, eta))

            manager.save()

            # get and save samples
            sample(step)




class LayoutNetDemo:
    def __init__(self, checkpoint_path):
        # define model
        self.layoutnet = layoutnet
        # fake input
        """[summary]

        Args:
            x ([type]): original layout annotation
            y ([type]): layout category
            tr (int64): text ratio
            ir (int64): image ratio
            img ([type]): image feature
            tex ([type]): text feature
            z ([type]): latent variable
            is_training (bool, optional): Training flag. Defaults to True.

        Returns:
            z_mean
            z_log_sigma_sq
            E
            G
            G_recon
            D_real
            D_fake
        """
        batch_size = config.batch_size
        height = 256
        width = 256
        channels = 3
        feature_dim = 128  # Adjust based on actual feature dimension expected by your model
        latent_dim = 100

        x = tf.random.normal([batch_size, height, width, channels])  # Original layout annotation
        y = tf.constant([1], dtype=tf.int64)  # Layout category
        tr = tf.constant([50], dtype=tf.int64)  # Text ratio
        ir = tf.constant([50], dtype=tf.int64)  # Image ratio
        img = tf.random.normal([batch_size, height, width, channels])  # Image feature as 4D tensor
        tex = tf.random.normal([batch_size, height, width, channels])
        z = tf.random.normal([batch_size, latent_dim])  # Latent variable

        '''
        z_mean, z_log_sigma_sq, E, G, G_recon, D_real, D_fake = self.layoutnet(
            x, y, tr, ir, img, tex, z, is_training=False
        )
        '''
        # load checkpoint


        # restore from latest checkpoint
        self.layoutnet.load_weights(checkpoint_path).expect_partial()
        
        train()

        #checkpoint = tf.train.Checkpoint(model=self.layoutnet)
        #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

        #self.layoutnet.summary()

        # Save the model in SavedModel format with the serving function


        # tf.saved_model.save(self.layoutnet, './saved_model')
        self.layoutnet.save('saved_modelh2', save_format='tf')

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
    demoModel = LayoutNetDemo(checkpoint_path="./checkpoints/ckpt-300")

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

        image_raw = demoModel.generate(
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
