import torch
import torch.nn as nn
import torch.optim as optim
from clip.model import LayerNorm
from models import tent

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Entropy(torch.nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        return -(x.softmax(0)*x.log_softmax(0)).sum(0).mean()


def setup_tent(model, type, lr, mode):
    """Set up tent adaptation.
    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    if mode == 'norm':
        model.visual = tent.configure_model(model.visual, type)
    #extractor = [model.net.conv1, model.net.bn1, nn.ReLU(inplace=True), model.net.layer1, model.net.layer2]
    #extractor = nn.Sequential(*extractor)
        params, param_names = tent.collect_params(model, type)
    elif mode == 'all':
        params = model.visual.parameters()
    elif mode == 'batch-adapter':
        adapter = nn.BatchNorm2d(3, 3)
        model.visual.adapter = adapter.cuda()
        params = model.visual.adapter.parameters()
    elif mode == 'conv-adapter':
        adapter = nn.Conv2d(3, 3, 1, bias=False)
        model.visual.adapter = adapter.cuda()
        params = model.visual.adapter.parameters()
    optimizer = setup_optimizer(params, lr)
    tent_model = tent.Tent(model, optimizer,
                           steps=10, ### Iterations
                           episodic=True)
    return tent_model

def setup_optimizer(params, lr):
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
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=0.0)

def test_forward(model, images, prompts, labels, prediction_true, prediction_false):
    with torch.no_grad():
        image_features = model.model.encode_image(images)
        text_features = model.model.encode_text(prompts)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, pred = similarity.topk(1, 1, True, True)
    pred = pred.t()
    correctness = pred.eq(labels.view(1, -1).expand_as(pred))

    acc3 = 0.0
    values3, pred3 = similarity.topk(3, 1, True, True)
    for k in range(len(labels)):
        if labels[k] in pred3:
            acc3 += 1
    for k in range(len(correctness)):
        if correctness[0, k] == True:
            prediction_true.append(100 * values[k].item())
        elif correctness[0, k] == False:
            prediction_false.append(100 * values[k].item())
    return acc3, correctness, prediction_true, prediction_false

def measure_lame(Y, labels):
    pred = Y.argmax(1)
    correctness = pred.eq(labels)
    acc3 = 0.0
    values3, pred3 = Y.topk(3, 1, True, True)
    for k in range(len(labels)):
        if labels[k] in pred3:
            acc3 += 1

    return acc3, correctness
