import os.path
import copy
import torch
import random
from PIL import Image
from tqdm import tqdm

import configuration, utils
from clip import clip
from dataset import prepare_dataset

classes = ['aeroplane', 'bycicle', 'bus', 'car', 'horse',
           'knife', 'motorcycle', 'person', 'plant',
           'skateboard', 'train', 'truck']

class_dict = {}
for i, cl in enumerate(classes):
    class_dict[i] = cl

def main(args):
    device = torch.device('cuda:0')
    if args.model in ['RN50', 'RN101']:
        type = 'resnet'
    else:
        type = 'vit'

    # Load model
    model, preprocess = clip.load(args.model, device)
    model = utils.setup_tent(model, type=type, lr=args.lr, mode=args.mode)

    #Dataset
    if args.split == 'train':
        trloader, trsampler, teloader, tesampler = prepare_dataset.prepare_train_data(args, transform=preprocess)
        trloader.dataset.classes = classes
        teloader.dataset.classes = classes
    else:
        trloader, trsampler = prepare_dataset.prepare_val_data(args, transform=preprocess)

    #Classification
    n_images = len(trloader.dataset)
    correct_before = 0
    correct_after = 0
    acc3_before = 0
    acc3_after = 0
    prediction_true_before = []
    prediction_true_after = []
    prediction_false_before = []
    prediction_false_after = []
    prompts = torch.cat([clip.tokenize(f'an photo of a {c}' for c in classes)]).to(device)
    for x, y in tqdm(trloader):
        x = x.to(device)
        y = y.to(device)
        model.reset()

        # Measure accuracy before adapting
        new_acc3, correctness, pred_true, pred_false = utils.test_forward(model, x, prompts, y, prediction_true_before, prediction_false_before)
        acc3_before += new_acc3
        correct_before += correctness.sum().item()

        if args.adapt:
            #Adapting
            Y = model(x, prompts, trloader.dataset, device, threshold_not=args.th, K=args.K, method=args.method, kernel=args.kernel)

            #Testing
            if args.method != 'lame':
                new_acc3, correctness, pred_true, pred_false = utils.test_forward(model, x, prompts, y, prediction_true_after, prediction_false_after)
            else:
                new_acc3, correctness = utils.measure_lame(Y, y)
            acc3_after += new_acc3
            correct_after += correctness.sum().item()


    #Printing results
    print('Model: ', args.model)
    print('Method:', args.method)
    print('Learning rate: ', args.lr)
    print('No. of classes: ', args.K)
    print('Threshold: ', args.th)
    accuracy_before = correct_before / n_images
    acc3_before = acc3_before / n_images
    print("Top-1 Accuracy before:", accuracy_before)
    print("Top-3 Accuracy:", acc3_before)
    if args.adapt:
        accuracy_after = correct_after / n_images
        acc3_after = acc3_after / n_images
        print("Top-1 Accuracy after:", accuracy_after)
        print("Top-3 Accuracy after:", acc3_after)


def test(model, x, y, prompts):
    with torch.no_grad():
        logits, logits_per_image, logits_per_text = model.model(x, prompts)
        probs = logits_per_image.softmax(dim=-1)
        pred = probs.argmax(dim=1)
        correct_predictions = (y == pred).sum().item()
        acc = correct_predictions / y.size(0)

    return acc


if __name__=='__main__':
    args = configuration.argparser()
    if args.livia:
        args.dataroot = '/export/livia/home/vision/gvargas/data/Visda/'
    main(args)


# Loading image
'''cl = 'horse'
root = os.path.join(args.dataroot, 'train', cl)
names = os.listdir(root)
image = Image.open(os.path.join(root, random.choice(names)))
image = preprocess(image).unsqueeze(0)
prompts = torch.cat([clip.tokenize(f'an image of a {c}' for c in classes)])'''