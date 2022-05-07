import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.distributions as dists
import numpy as np

from netcal.metrics import ECE
from laplace import Laplace
from netcal.scaling import TemperatureScaling

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from bert_utils import load_model_and_data, parse_cmd_args
from bert_wrapper import BertMAP

@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x))
        else:
            py.append(torch.softmax(model(x), dim=-1))
    return torch.cat(py).cpu()

# First evaluate calibration of the pre-trained model
def load_all():
    model_args, data_args, training_args = parse_cmd_args()

    # print(model_args, data_args, training_args)

    tokenizer, config, model, dataset_dict = load_model_and_data(model_args, data_args, training_args)
    
    # print(model)

    # print(type(dataset_dict['train']))

    # print(len(dataset_dict['train']))
    # print(len(dataset_dict['eval']))

    # for x, y in dataset_dict['train']:
    #     print(x)
    #     print(y)

    return tokenizer, config, model, dataset_dict

if __name__=="__main__":

    tokenizer, config, model, dataset_dict = load_all()

    bert_model = BertMAP(model)

    # for name, layer in bert_model.named_modules():
        # print(name)
    train_loader = dataset_dict['train']
    eval_loader = dataset_dict['eval']
    targets = torch.cat([y for x, y in eval_loader], dim=0).cpu()
    probs_map = predict(eval_loader, bert_model, laplace=False)
    acc_map = (probs_map.argmax(-1) == targets).float().mean()
    
    if probs_map.shape[-1] == 2: # binary classification
        class_prob = torch.gather(probs_map, 1, targets.view(-1, 1))
        ece_map = ECE(bins=15).measure(class_prob.numpy(), targets.numpy())

    print(acc_map)
    print(probs_map.numpy().shape, targets.numpy().shape)
    # ece_map = ECE(bins=10).measure(probs_map.numpy(), targets.numpy())
    # ece_map = 0.0
    nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
    print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')

    # Temperature Scaling
    temperature = TemperatureScaling()
    temperature.fit(probs_map, targets)
    calibrated_probs_map = temperature.transform(probs_map)
    if calibrated_probs_map.shape[-1] == 2: # binary classification
        class_prob = torch.gather(calibrated_probs_map, 1, targets.view(-1, 1))
        ece_map = ECE(bins=15).measure(class_prob.numpy(), targets.numpy())
    print('New ECE ', ece_map)
