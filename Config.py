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
        self.logdir = "./log"
        self.sampledir = "./example"
        self.max_steps = 1
        self.sample_every_n_steps = 100
        self.summary_every_n_steps = 1
        self.save_model_secs = 120
        self.checkpoint_basename = "layout"
        self.checkpoint_dir = "./checkpoints"
        self.filenamequeue = "./dataset/layout_1205.tfrecords"
        self.min_after_dequeue = 5000
        self.num_threads = 4


config = Config()
