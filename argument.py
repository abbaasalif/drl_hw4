def add_arguments(parser):
    '''
    Add your additional arguments here if needed. The TAs will run main.py --test_* to load
    your default arguments.

    '''
    parser.add_argument('--batch_size','-b', type = int, default = 32, help = 'batch size for training')
    parser.add_argument('--episode', '-ep', type = int, default = 1000, help = "how many episodes for training")
    parser.add_argument('--learning_rate','-lr', type = float, default = 0.0001, help = 'the learning rate')
    parser.add_argument('--gamma', type = float, default = 0.99, help = 'the discount factor of reward')
    parser.add_argument('--optim', type = str, default = "Adam", help = "the optimizer: Adam, RMSprop, SGD")
    parser.add_argument('--model_name', type = str, default = "default_model.ckpt", help = "the name of model for training or test")

    return parser
