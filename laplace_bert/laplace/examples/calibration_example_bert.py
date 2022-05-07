import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import torch.distributions as dists
import numpy as np

from netcal.metrics import ECE
from laplace import Laplace
from utils import _ECELoss

np.random.seed(7777)
torch.manual_seed(7777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

from bert_utils import load_model_and_data, parse_cmd_args
from bert_wrapper import BertMAP

@torch.no_grad()
def predict(dataloader, model, laplace=False, only_logits=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x))
        else:
            if only_logits:
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
    # logits_map = predict(eval_loader, bert_model, laplace=False, only_logits=True)
    acc_map = (probs_map.argmax(-1) == targets).float().mean()

    if probs_map.shape[-1] == 2: # binary classification
        class_prob = torch.gather(probs_map, 1, targets.view(-1, 1))
        ece_map = ECE(bins=15).measure(class_prob.numpy(), targets.numpy())
    else:
        ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())

    ece_map_2 = _ECELoss().forward(probs_map, targets, are_probs=True)
    print('ECE 1 ', ece_map)
    print('ECE 2 ', ece_map_2)

    print(acc_map)
    print(probs_map.numpy().shape, targets.numpy().shape)
    # ece_map = ECE(bins=10).measure(probs_map.numpy(), targets.numpy())
    # ece_map = 0.0
    nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()
    print(f'[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}')

    # Laplace
    la = Laplace(bert_model, 'classification',
                subset_of_weights='last_layer',
                last_layer_name='bert_model.classifier',
                hessian_structure='full',
                )
    la.fit(train_loader)
    la.optimize_prior_precision(method='marglik')

    probs_laplace = predict(eval_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
    if probs_laplace.shape[-1] == 2: # binary classification
        class_prob_laplace = torch.gather(probs_laplace, 1, targets.view(-1, 1))
        ece_laplace = ECE(bins=15).measure(class_prob_laplace.numpy(), targets.numpy())
    else:
        ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    ece_laplace_2 = _ECELoss().forward(probs_laplace, targets, are_probs=True)
    nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

    print('[Laplace] ECE - 2 ', ece_laplace_2)
    print(f'[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}')
