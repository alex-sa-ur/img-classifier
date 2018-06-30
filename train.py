import argparse
import image_classifier
import utils

data_path       = 'flowers'
save_path       = ''
predict_path    = 'flowers/test/1/image_06743.jpg'
arch_type       = 'vgg'
learn_rate      = 0.0001
hidden_units    = 1024
epoch_number    = 3
gpu_usage       = False
top_k           = 5
category_names  = 'cat_to_name.json'

def pass_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory', action='store',
                        type=str, 
                        help='path to training folder'
    )
    
    parser.add_argument('--save_dir',action='store',
                        type=str, default=save_path, 
                        help='path to save checkpoint. [DEFAULT: \'{}\']'.format(save_path)
    )
    
    parser.add_argument('--arch', action='store',
                        type=str, default=arch_type, choices=set(('vgg','densenet')),
                        help='model type to be used. [DEFAULT: \'{}\']'.format(arch_type)
    )
    
    parser.add_argument('--learning_rate', action='store',
                        type=float, default=learn_rate, 
                        help='learning rate for training. [DEFAULT: {}]'.format(learn_rate)
    )
    
    parser.add_argument('--hidden_units', action='store',
                        type=int, default=hidden_units,
                        help='units desired in hidden layer. [DEFAULT: {}]'.format(hidden_units)
    )
    
    parser.add_argument('--epochs', action='store',
                        type=int, default=epoch_number,
                        help='epochs desired for training. [DEFAULT: {}]'.format(epoch_number)
    )
    
    parser.add_argument('--gpu', action='store_true', 
                        default=gpu_usage,
                        help='use GPU for training'
    )
    
    return parser.parse_args()
    
def main():
    args = pass_args()
    data_path       = args.data_directory
    save_path       = args.save_dir
    arch_type       = args.arch
    learn_rate      = args.learning_rate
    hidden_units    = args.hidden_units
    epoch_number    = args.epochs
    gpu_usage       = args.gpu
    
    classifier_model = image_classifier.model(
        data_path,
        save_path,
        arch_type,
        learn_rate, 
        hidden_units, 
        epoch_number, 
        gpu_usage,
        True,
        predict_path,
        top_k,
        category_names
    )
    
    classifier_model.train()
    classifier_model.save_checkpoint()
    
if __name__ == '__main__':
    main()