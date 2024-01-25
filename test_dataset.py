import os

import numpy as np

import clip
import torch
from tqdm import tqdm
from torchvision.datasets import CIFAR100

import configuration
from utils import prepare_dataset, utils

# Argues
args = configuration.argparser()

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

common_corruptions = ['original', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
fichier = open(args.dataset+'output.txt', 'w')
for cor in common_corruptions:
    args.corruption = cor

    # Download the dataset
    teloader, _, teset = prepare_dataset.prepare_test_data(args)

    correct = 0
    acc3 = 0
    prediction_true = []
    prediction_false = []
    for batch_idx, (inputs, labels) in tqdm(enumerate(teloader)):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in teset.classes]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(inputs)
            text_features = model.encode_text(text_inputs)

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
    print('Corruption:', cor)
    print("Accuracy:", accuracy)
    print("Accuracy3:", acc3)
    print("Confidence of prediction when it's True:", np.mean(prediction_true), "+/-", np.std(prediction_true), "min:", prediction_true.min(), "max:", prediction_true.max())
    print("Confidence of prediction when it's False:", np.mean(prediction_false), "+/-", np.std(prediction_false),  "min:", prediction_false.min(), "max:", prediction_false.max())
    ecrit = str(round(accuracy*100,2))+ ','+str(round(acc3*100,2))+','+str(round(np.mean(prediction_true),2))+','+str(round(np.std(prediction_true),2)) + ','+str(round(prediction_true.min(),2))+ ','+str(round(prediction_true.max(),2)) + ','+str(round(np.mean(prediction_false),2))+','+str(round(np.std(prediction_false),2)) + ','+str(round(prediction_false.min(),2))+ ','+str(round(prediction_false.max(),2))+'\n'
    fichier.write(ecrit)