import os

import numpy as np

import clip
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision.datasets import CIFAR100
import logging

import configuration, tent
from utils import prepare_dataset, utils


def setup_tent(model):
    """Set up tent adaptation.
    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model.visual = tent.configure_model(model.visual)
    #extractor = [model.net.conv1, model.net.bn1, nn.ReLU(inplace=True), model.net.layer1, model.net.layer2]
    #extractor = nn.Sequential(*extractor)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=10, ### Iterations
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
base_model, preprocess = clip.load('ViT-B/32', device)
model = setup_tent(base_model)

Confidence_th = [1.0, 0.8, 0.6, 0.4]
Batch_size = [128, 64, 32, 4, 2, 1]
fichier = open(str(args.K)+'_K.txt', 'w')
for bs in Batch_size:
    Ecrit = ''
    args.batch_size = bs
    # Download the dataset
    teloader, _, teset = prepare_dataset.prepare_test_data(args)
    for th in Confidence_th:
        args.threshold_not = th
        correct = 0
        acc3 = 0
        prediction_true = []
        prediction_false = []
        for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in teset.classes]).to(device)
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")

            _ = model(inputs, text_inputs, teset, device, threshold_not = args.threshold_not, K = args.K)  # infer and adapt

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
            pred = pred.t()
            correctness = pred.eq(labels.view(1, -1).expand_as(pred))

            values3, pred3 = similarity.topk(3, 1, True, True)
            for k in range(len(labels)):
                if labels[k] in pred3:
                    acc3 += 1
            for k in range(len(correctness)):
                if correctness[0, k] == True:
                    prediction_true.append(100 * values[k].item())
                elif correctness[0, k] == False:
                    prediction_false.append(100 * values[k].item())
            correct += correctness.sum().item()
            #acc = utils.accuracy(similarity, labels)

        accuracy = correct / len(teloader.dataset)
        acc3 = acc3/ len(teloader.dataset)
        prediction_true = np.array(prediction_true)
        prediction_false = np.array(prediction_false)
        # Print the result
        print('Corruption:', args.corruption)
        print("Accuracy:", accuracy)
        print("Accuracy3:", acc3)
        Ecrit = Ecrit + str(round(accuracy*100,2)) + ','
    fichier.write(Ecrit + '\n')