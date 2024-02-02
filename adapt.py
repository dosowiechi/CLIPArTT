import numpy as np

import clip
import torch
import torch.optim as optim
from tqdm import tqdm
import logging

import configuration
from models import tent, lame
from utils import prepare_dataset


def setup_tent(model, name_model, niter = 10, method = 'clip'):
    """Set up tent adaptation.
    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    LN = True
    if LN == True:
        model.visual = tent.configure_model(model.visual, name_model)
        #extractor = [model.net.conv1, model.net.bn1, nn.ReLU(inplace=True), model.net.layer1, model.net.layer2]
        #extractor = nn.Sequential(*extractor)
        params, param_names = tent.collect_params(model.visual, name_model)
    else:
        params = model.visual.parameters()
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=niter,  ### Iterations
                           method=method,
                           episodic=True)
    return tent_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.
    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.
    For best results, try tuning the learning rate and batch size.
    """
    # if cfg.OPTIM.METHOD == 'Adam':
    return optim.Adam(params,
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=0.0)

# Argues
args = configuration.argparser()
logger = logging.getLogger(__name__)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model, preprocess = clip.load(args.model, device)
model = setup_tent(base_model, args.model, niter=args.niter, method = args.method)

common_corruptions = [args.corruption]
# common_corruptions = ['cifar_new'] #'original', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      #'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      #'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
# fichier = open('Results/' + args.dataset + '_' + args.model.replace('/','') + '_niter' + str(args.niter) + '_K' + str(args.K)+'.txt', 'w')
fichier = open('Results/' + args.dataset + '_' + args.model.replace('/','') + '.txt', 'w')
for cor in common_corruptions:
    # Ecrit = ''
    args.corruption = cor
    validation = 3
    # Download the dataset
    teloader, _, teset = prepare_dataset.prepare_test_data(args)
    if cor == 'cifar_new':
        args.corruption = 'original'
        _, _, teset = prepare_dataset.prepare_test_data(args)
    acc = []
    for _ in range(validation):
        correct = 0
        for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in teset.classes]).to(device)

            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
            if args.adapt:

                Y = model(inputs, text_inputs, teset, device, threshold_not = args.threshold_not, K = args.K)  # infer and adapt

            if args.method in ['clipartt', 'tent'] or not args.adapt:
                # Calculate features
                with torch.no_grad():
                    image_features = model.model.encode_image(inputs)
                    text_features = model.model.encode_text(text_inputs)

                # Pick the top 5 most similar labels for the image
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                #values, indices = similarity.topk(5)
                values, pred = similarity.topk(1, 1, True, True)
            elif args.method == 'lame':
                pred = Y.argmax(1)
                pred = pred.unsqueeze(1)
            pred = pred.t()
            correctness = pred.eq(labels.view(1, -1).expand_as(pred))
            correct += correctness.sum().item()

        acc.append(correct / len(teloader.dataset))
    print(str(round(np.array(acc).mean()*100,2)) + ',' + str(round(np.array(acc).std()*100,2)))
    fichier.write(str(round(np.array(acc).mean()*100,2)) + ',' + str(round(np.array(acc).std()*100,2)) + '\n')