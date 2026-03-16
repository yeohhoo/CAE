from argparse import ArgumentParser
from configs.paths_config import model_paths
from CAE.models.methods import backbone


model_dict = dict(
    Conv4=backbone.Conv4,
    Conv4S=backbone.Conv4S,
    Conv6=backbone.Conv6,
    ResNet10=backbone.ResNet10,
    ResNet18=backbone.ResNet18,
    ResNet34=backbone.ResNet34,
    ResNet50=backbone.ResNet50,
    ResNet101=backbone.ResNet101,
)


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument(
            '--exp_dir', type=str, help='Path to experiment output directory'
        )
        self.parser.add_argument(
            '--dataset_type',
            default='ffhq_encode',
            type=str,
            help='Type of dataset/experiment to run',
        )
        self.parser.add_argument(
            '--encoder_type',
            default='GradualStyleEncoder',
            type=str,
            help='Which encoder to use',
        )
        self.parser.add_argument(
            '--input_nc',
            default=3,
            type=int,
            help='Number of input image channels to the psp encoder',
        )
        self.parser.add_argument(
            '--label_nc',
            default=0,
            type=int,
            help='Number of input label channels to the psp encoder',
        )
        self.parser.add_argument(
            '--feature_size',
            default=256,
            type=int,
            help='Dimension of latent code in hyperbolic space',
        )
        self.parser.add_argument(
            '--output_size',
            default=1024,
            type=int,
            help='Output size of generator',
        )
        self.parser.add_argument(
            '--d_output_size',
            default=256,
            type=int,
            help='Output size of generator',
        )

        # for metric learning
        self.parser.add_argument(
            '--train_n_way',
            default=5,
            type=int,
            help='class num to classify for training',
        )  # baseline and baseline++ would ignore this parameter
        self.parser.add_argument(
            '--test_n_way',
            default=5,
            type=int,
            help='class num to classify for testing (validation) ',
        )  # baseline and baseline++ only use this parameter in finetuning
        self.parser.add_argument(
            '--n_shot',
            default=1,
            type=int,
            help='number of labeled data in each class, same as n_support',
        )  # baseline and baseline++ only use this parameter in finetuning
        self.parser.add_argument(
            '--train_aug',
            action='store_true',
            help='perform data augmentation or not during training ',
        )  # still required for save_features.py and test.py to find the model path correctly
        self.parser.add_argument(
            '--model',
            default='Conv4',
            help='model: Conv{4|6} / ResNet{10|18|34|50|101}',
        )  # 50 and 101 are not used in the paper
        self.parser.add_argument(
            '--method',
            default='baseline',
            help='baseline/baseline++/protonet/protonet_PSM/matchingnet/relationnet_PSM/relationnet{_softmax}/maml{_approx}',
        )  # relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
        self.parser.add_argument(
            '--n_eposide',
            default=2200,
            type=int,
            help='2200(102flowers)/baseline++/protonet/protonet_PSM/matchingnet/relationnet_PSM/relationnet{_softmax}/maml{_approx}',
        )  # relationnet_softmax replace L2 norm with softmax to expedite training, maml_approx use first-order approximation in the gradient for efficiency
        self.parser.add_argument(
            '--metric_file',
            default=None,
            type=str,
            help='Path to metric model checkpoint',
        )

        self.parser.add_argument(
            '--vgg_lambda',
            default=0.05,
            type=float,
            help='LPIPS loss multiplier factor',
        )
        self.parser.add_argument(
            '--adv_lambda',
            default=0.1,
            type=float,
            help='LPIPS loss multiplier factor',
        )
        self.parser.add_argument(
            '--metric_lambda',
            default=10.0,
            type=float,
            help='LPIPS loss multiplier factor',
        )

        self.parser.add_argument(
            '--batch_size', default=4, type=int, help='Batch size for training'
        )
        self.parser.add_argument(
            '--test_batch_size',
            default=2,
            type=int,
            help='Batch size for testing and inference',
        )
        self.parser.add_argument(
            '--workers',
            default=4,
            type=int,
            help='Number of train dataloader workers',
        )
        self.parser.add_argument(
            '--test_workers',
            default=2,
            type=int,
            help='Number of test/inference dataloader workers',
        )

        self.parser.add_argument(
            '--learning_rate',
            default=0.0001,
            type=float,
            help='Optimizer learning rate',
        )
        self.parser.add_argument(
            '--learning_rate_d',
            default=0.0001,
            type=float,
            help='Optimizer learning rate',
        )
        self.parser.add_argument(
            '--optim_name',
            default='ranger',
            type=str,
            help='Which optimizer to use',
        )
        self.parser.add_argument(
            '--train_decoder',
            default=False,
            type=bool,
            help='Whether to train the decoder model',
        )
        self.parser.add_argument(
            '--start_from_latent_avg',
            action='store_true',
            help='Whether to add average latent vector to generate codes from encoder.',
        )
        self.parser.add_argument(
            '--learn_in_w',
            action='store_true',
            help='Whether to learn in w space instead of w+',
        )
        self.parser.add_argument(
            '--d_reg_every',
            type=int,
            default=16,
            help="interval of the applying r1 regularization",
        )
        self.parser.add_argument(
            "--r1",
            type=float,
            default=10,
            help="weight of the r1 regularization",
        )
        self.parser.add_argument(
            '--e_reg_every',
            type=int,
            default=4,
            help="interval of the applying path length regularization",
        )
        self.parser.add_argument(
            "--path_batch_shrink",
            type=int,
            default=2,
            help="batch size reducing factor for the path length regularization (reduce memory consumption)",
        )
        self.parser.add_argument(
            "--path_regularize",
            type=float,
            default=2,
            help="weight of the path length regularization",
        )

        self.parser.add_argument(
            '--lpips_lambda',
            default=0.8,
            type=float,
            help='LPIPS loss multiplier factor',
        )
        self.parser.add_argument(
            '--id_lambda',
            default=0,
            type=float,
            help='ID loss multiplier factor',
        )
        self.parser.add_argument(
            '--l2_lambda',
            default=1.0,
            type=float,
            help='L2 loss multiplier factor',
        )
        self.parser.add_argument(
            '--w_norm_lambda',
            default=0,
            type=float,
            help='W-norm loss multiplier factor',
        )
        self.parser.add_argument(
            '--lpips_lambda_crop',
            default=0,
            type=float,
            help='LPIPS loss multiplier factor for inner image region',
        )
        self.parser.add_argument(
            '--l2_lambda_crop',
            default=0,
            type=float,
            help='L2 loss multiplier factor for inner image region',
        )
        self.parser.add_argument(
            '--moco_lambda',
            default=0,
            type=float,
            help='Moco-based feature similarity loss multiplier factor',
        )
        self.parser.add_argument(
            '--contrastive_lambda',
            default=0.1,
            type=float,
            help='Un/Supervised Contrastive loss factor for hyperbolic feature learning',
        )
        self.parser.add_argument(
            '--hyperbolic_lambda',
            default=0.1,
            type=float,
            help='Supervised hyperbolic loss for hyperbolic feature learning',
        )
        self.parser.add_argument(
            '--reverse_lambda',
            default=0.1,
            type=float,
            help='Project back to the original Euclidean space',
        )

        self.parser.add_argument(
            '--stylegan_weights',
            default=model_paths['stylegan_ffhq'],
            type=str,
            help='Path to StyleGAN model weights',
        )
        self.parser.add_argument(
            '--checkpoint_path',
            default=None,
            type=str,
            help='Path to HAE model checkpoint',
        )
        self.parser.add_argument(
            '--psp_checkpoint_path',
            default=None,
            type=str,
            help='Path to pSp model checkpoint',
        )
        self.parser.add_argument(
            '--gan_checkpoint_path',
            default=None,
            type=str,
            help='Path to Stylegan model checkpoint',
        )

        self.parser.add_argument(
            '--max_steps',
            default=500000,
            type=int,
            help='Maximum number of training steps',
        )
        self.parser.add_argument(
            '--image_interval',
            default=100,
            type=int,
            help='Interval for logging train images during training',
        )
        self.parser.add_argument(
            '--board_interval',
            default=50,
            type=int,
            help='Interval for logging metrics to tensorboard',
        )
        self.parser.add_argument(
            '--val_interval',
            default=1000,
            type=int,
            help='Validation interval',
        )
        self.parser.add_argument(
            '--save_interval',
            default=None,
            type=int,
            help='Model checkpoint interval',
        )

        # arguments for weights & biases support
        self.parser.add_argument(
            '--use_wandb',
            action="store_true",
            help='Whether to use Weights & Biases to track experiment.',
        )

        # arguments for super-resolution
        self.parser.add_argument(
            '--resize_factors',
            type=str,
            default=None,
            help='For super-res, comma-separated resize factors to use for inference.',
        )

        # invert.py
        self.parser.add_argument(
            '--image_list', type=str, help='List of images to invert.'
        )
        # self.parser.add_argument('-o', '--output_dir', type=str, default='', help='Directory to save the results. If not specified, '
        # 						'`./results/inversion/${IMAGE_LIST}` ''will be used by default.')
        self.parser.add_argument(
            '--p_learning_rate',
            type=float,
            default=0.01,
            help='Learning rate for optimization. (default: 0.01)',
        )
        self.parser.add_argument(
            '--num_iterations',
            type=int,
            default=100,
            help='Number of optimization iterations. (default: 100)',
        )
        self.parser.add_argument(
            '--num_results',
            type=int,
            default=5,
            help='Number of intermediate optimization results to '
            'save for each sample. (default: 5)',
        )
        self.parser.add_argument(
            '--loss_pix_local',
            type=float,
            default=5.0,
            help='The perceptual loss scale for optimization. ' '(default: 5)',
        )
        self.parser.add_argument(
            '--loss_weight_pix',
            type=float,
            default=1.0,
            help='The perceptual loss scale for optimization. '
            '(default: 5e-5)',
        )
        self.parser.add_argument(
            '--loss_hf_weight',
            type=float,
            default=1.0,
            help='The perceptual loss scale for optimization. '
            '(default: 5e-5)',
        )
        self.parser.add_argument(
            '--loss_weight_feat',
            type=float,
            default=5e-5,
            help='The perceptual loss scale for optimization. '
            '(default: 5e-5)',
        )
        self.parser.add_argument(
            '--loss_weight_enc',
            type=float,
            default=2.0,
            help='The encoder loss scale for optimization.' '(default: 2.0)',
        )
        self.parser.add_argument(
            '--viz_size',
            type=int,
            default=256,
            help='Image size for visualization. (default: 256)',
        )
        self.parser.add_argument(
            '--gpu_id',
            type=str,
            default='0',
            help='Which GPU(s) to use. (default: `0`)',
        )

        # compute jacobian
        self.parser.add_argument(
            '--latent_path',
            type=str,
            default='',
            help='Path to the given latent codes. (default: None)',
        )
        self.parser.add_argument(
            '--latent_label_path',
            type=str,
            default='',
            help='Path to the given labels of latent codes. (default: None)',
        )
        self.parser.add_argument(
            '--seed',
            type=int,
            default=4,
            help='Seed for sampling. (default: 4)',
        )
        self.parser.add_argument(
            '--nums',
            type=int,
            default=5,
            help='Number of samples to synthesized. (default: 5)',
        )
        self.parser.add_argument(
            '--data_name',
            type=str,
            default='102flower',
            help='Name of the datasets. (default: ffhq)',
        )
        self.parser.add_argument(
            '--save_dir',
            type=str,
            default=None,
            help='Directory to save the results. If not specified, '
            'the results will be saved to `work_dirs/{TASK_SPECIFIC}/` by default.',
        )
        self.parser.add_argument(
            '--save_jpg',
            action='store_false',
            help='Whether to save the images used to compute '
            'jacobians. (default: True)',
        )

        # compute direction
        self.parser.add_argument(
            '--jaco_path',
            type=str,
            default=None,
            help='Directory to save the results. If not specified, '
            'the results will be saved to `work_dirs/{TASK_SPECIFIC}/` by default.',
        )
        self.parser.add_argument(
            '--region',
            type=str,
            default='full',
            help='The region to be used to compute jacobian.',
        )
        self.parser.add_argument(
            '--full_rank',
            action='store_true',
            help='Whether or not to full rank background(default: False).',
        )
        self.parser.add_argument(
            '--tao',
            type=float,
            default=0.01,
            help='The rank for computing directions. ' '(default: 0.01)',
        )
        # manipulate
        self.parser.add_argument(
            '--boundary_path', type=str, help='Path to the attribute vectors.'
        )
        self.parser.add_argument(
            '--mani_layers',
            type=str,
            default='4,5,6,7',
            help='The layers will be manipulated.'
            '(default: 4,5,6,7). For the eyebrow and lipstick,'
            'using [8-11] layers instead.',
        )
        self.parser.add_argument(
            '--vis_size',
            type=int,
            default=256,
            help='Size of the visualize images. (default: 256)',
        )
        self.parser.add_argument(
            '--step',
            type=int,
            default=7,
            help='Number of manipulation steps. (default: 7)',
        )
        self.parser.add_argument(
            '--start',
            type=int,
            default=0,
            help='The start index of the manipulation directions.',
        )
        self.parser.add_argument(
            '--end',
            type=int,
            default=1,
            help='The end index of the manipulation directions.',
        )
        self.parser.add_argument(
            '--start_distance',
            type=float,
            default=-10.0,
            help='Start distance for manipulation. (default: -10.0)',
        )
        self.parser.add_argument(
            '--end_distance',
            type=float,
            default=10.0,
            help='End distance for manipulation. (default: 10.0)',
        )

        # generate refine training pairs
        self.parser.add_argument(
            '--or_root',
            type=str,
            default='',
            help='Path to the orignal root. (default: None)',
        )
        self.parser.add_argument(
            '--save_root',
            type=str,
            default='',
            help='Path to the save root. (default: None)',
        )
        self.parser.add_argument(
            '--job',
            type=str,
            default='Inversion',
            help="which function ('Inversion' or 'Manipulate')to run in inference",
        )

    def parse(self):
        opts = self.parser.parse_args()
        return opts
