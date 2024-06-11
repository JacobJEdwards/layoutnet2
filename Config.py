class Config:
    def __init__(self):
        # configuration for building the network
        self.y_dim = 6
        self.tr_dim = 7
        self.ir_dim = 10
        self.latent_dim = 128
        self.z_dim = 128
        self.batch_size = 128
        self.lr = 0.0002
        self.beta1 = 0.5
        # configuration for the supervisor
        self.logdir = "/gdrive/My Drive/LayoutNet/log"
        self.sampledir = "/gdrive/My Drive/LayoutNet/example"
        self.max_steps = 30000
        self.sample_every_n_steps = 100
        self.summary_every_n_steps = 1
        self.save_model_secs = 120
        self.checkpoint_basename = "layout"
        self.checkpoint_dir = "/gdrive/My Drive/LayoutNet/checkpoints"
        self.filenamequeue = "/gdrive/My Drive/LayoutNet/dataset/layout_1205.tfrecords"
        self.min_after_dequeue = 5000
        self.num_threads = 4


config = Config()
