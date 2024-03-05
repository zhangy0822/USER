import os
import argparse
import logging
from lib import evaluation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco',
                        help='coco or f30k')
    parser.add_argument('--data_path', default='your dataset path')
    parser.add_argument('--trained_time', default='your pretrained folder name ')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--evaluate_cxc', action='store_true')
    opt = parser.parse_args()
    
    if opt.dataset == 'coco':
        weights_bases = [
            './runs/coco',
        ]
    elif opt.dataset == 'f30k':
        weights_bases = [
            './runs/f30k',
        ]
    else:
        raise ValueError('Invalid dataset argument {}'.format(opt.dataset))

    for base in weights_bases:
        model_path = os.path.join(base, opt.trained_time, 'checkpoints/model_best.pth')
        logger.info('Evaluating {}...'.format(model_path))
        if opt.save_results:  # Save the final results for computing ensemble results
            save_path = os.path.join(base, opt.trained_time, 'results_{}.npy'.format(opt.dataset))
        else:
            save_path = None

        if opt.dataset == 'coco':
            if not opt.evaluate_cxc:
                # Evaluate COCO 5-fold 1K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
                # Evaluate COCO 5K
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
            else:
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True, cxc=True)
        elif opt.dataset == 'f30k':
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)


if __name__ == '__main__':
    main()
