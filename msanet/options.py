import argparse

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Object Counting Framework")
    # project name
    parser.add_argument('--project_name', default='soy-test')

    # constant
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=[], type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--beta', default=0.9, type=float)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # a threshold during evaluation for counting and visualization
    parser.add_argument('--threshold', default=0.1, type=float,
                        help="threshold in evalluation: evaluate_crowd_no_overlap")

    # dataset parameters
    parser.add_argument('--dataset_file', default='Soy')
    parser.add_argument('--data_root', default='../data/2021_dataset',
                        help='path where the dataset is')
    
    parser.add_argument('--output_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=10, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--vis_freq', default=5, type=int,
                        help='frequency of visualization, default setting is visualize in every 5 epoch')
    parser.add_argument('--save_freq', default=100, type=int,
                        help='frequency of saving checkpoint, default setting is saving in every 100 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    #
    opt = parser.parse_known_args()[0]
    return opt