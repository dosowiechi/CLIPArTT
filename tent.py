from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

import clip


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, method='clip', episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.method = method

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, text_x, teset, device, threshold = 1, threshold_not = 1, K=3):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            #forward_and_adapt(x, text_x, teset, device, self.model, self.optimizer, threshold=0,
            #                  threshold_not=1, K=K)
            forward_and_adapt(x, text_x, teset, device, self.model, self.optimizer, method = self.method, threshold = threshold, threshold_not = threshold_not, K=K)

        return 0

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def getprompt(K, c, teset):
    for k in range(K):
        if k == 0:
            text_prompt = f"a photo of a " + teset.classes[c[k]]
        else:
            text_prompt = text_prompt + " or " + teset.classes[c[k]]
    return text_prompt


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, text_x, teset, device, model, optimizer, method = 'clip', threshold = 1, threshold_not = 1, K=3):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    if method == 'clip':
        # forward
        with torch.no_grad():
            image_features = model.encode_image(x)
            text_features = model.encode_text(text_x)
        # adapt
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, pred = similarity.topk(K, 1, True, True)
        confidence = values[:,0] > threshold
        not_confidence = values[:,0] < threshold_not
        pred_conf = pred[:,0][confidence].int()
        if len(pred_conf) != 0:
            pred_inputs = torch.cat([clip.tokenize(f"a photo of a {teset.classes[c]}") for c in pred_conf]).to(device)

        pred_notconf = pred[not_confidence]
        if len(pred_notconf) != 0:
            pred_inputs_not = torch.cat(
                [clip.tokenize(getprompt(K, c, teset)) for
                 c in pred_notconf]).to(device)

        if len(pred_conf) == 0 and len(pred_notconf) == 0:
            return 0
        elif len(pred_conf) == 0:
            x_new = x[not_confidence]
            pred_new = pred_inputs_not
        elif len(pred_notconf) == 0:
            x_new = x[confidence]
            pred_new = pred_inputs
            # return 0
        else:
            x_new = torch.cat([x[confidence],x[not_confidence]],0)
            pred_new = torch.cat([pred_inputs,pred_inputs_not],0)
        # Calculating the Loss
        # cosine similarity as logits
        logits, image_features, text_features=model(x_new, pred_new)
        images_similarity = image_features @ image_features.t()
        texts_similarity = text_features @ text_features.t()
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 , dim=-1
        )
        if len(pred_conf) == 0:
            loss = -cross_entropy(logits.t(), targets.t(), reduction='mean')
        else:
            loss = -cross_entropy(logits.t(), targets.t(), reduction='mean')
    elif method == 'tent':
        # forward
        logits, image_features, text_features = model(x, text_x)
        # adapt
        loss = softmax_entropy(logits).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def collect_params(model, name_model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if 'RN' in name_model:
            if isinstance(m, nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        else:
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, name_model):
    """Configure model for use with tent."""
    if 'RN' in name_model:
        # train mode, because tent optimizes the model to minimize entropy
        model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if 'RN' in name_model:
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        else:
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"