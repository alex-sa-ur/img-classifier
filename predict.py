import argparse
import image_classifier
import json
import utils

data_path       = 'flowers'
save_path       = 'checkpoint.pth'
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
    
    parser.add_argument('input', action='store',
                        type=str, 
                        help='path to predicting image'
    )
    
    parser.add_argument('checkpoint', action='store',
                        type=str, 
                        help='path to checkpoint'
    )
    
    parser.add_argument('--top_k', action='store',
                        type=int, default=top_k,
                        help='number of top probabilities to show. [DEFAULT: \'{}\']'.format(top_k)
    )
    
    parser.add_argument('--category_names', action='store',
                        type=str, default=category_names, 
                        help='path to file containing names. [DEFAULT: {}]'.format(category_names)
    )
    
    parser.add_argument('--gpu', action='store_true', 
                        default=gpu_usage,
                        help='use GPU for training'
    )
    
    return parser.parse_args()

def main():
    args = pass_args()
    predict_path    = args.input
    save_path       = args.checkpoint
    top_k           = args.top_k
    category_names  = args.category_names
    gpu_usage       = args.gpu
    
    classifier_model = image_classifier.model(
        data_path,
        save_path,
        arch_type,
        learn_rate, 
        hidden_units, 
        epoch_number, 
        gpu_usage,
        False,
        predict_path,
        top_k,
        category_names
    )
    
    classifier_model.load_checkpoint()
    classifier_model.predict()
    
if __name__ == '__main__':
    main()