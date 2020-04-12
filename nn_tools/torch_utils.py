import math
from copy import deepcopy
from typing import List

import torch

from .utils import tqdm


def forever_iter(iterable):
    while True:
        for _ in iterable:
            yield _


def evaluate(module: torch.nn.Module, dataloader, loss: list, ncols=None):
    with torch.no_grad():
        module.eval()
        generator = forever_iter(dataloader)
        loss_value = [[] for _ in loss]
        for _ in tqdm(range(len(dataloader)), ncols=ncols):
            data, label = next(generator)
            prediction = module(*data)
            for a, b in zip(loss_value, loss):
                a.append(b(*prediction, *label))
        loss_value = torch.tensor(loss_value, device=loss_value[0][0].device).mean(dim=-1)
    return loss_value


def fit(module: torch.nn.Module, train_dataloader, valid_data_loader, optimizer, scheduler, max_step, loss: list,
        is_higher_better, metric_loss_id=-1, early_stopping_rounds=-1, verbose=1, metric_value=None, ncols=None):
    # 状态变量
    print('using {} as training loss, using {} as early stopping metric'.format(
        type(loss[0]).__name__, type(loss[metric_loss_id]).__name__))
    best_state_dict = deepcopy(module.state_dict())
    best_step = -1
    best_metric_value = metric_value
    loss_record = []
    try:
        step = 0
        generator = forever_iter(train_dataloader)
        while step < max_step:
            for _ in tqdm(range(verbose), ncols=ncols):
                step += 1
                # --------- 训练参数 ------------
                module.train(True)
                data, label = next(generator)
                optimizer.zero_grad()
                prediction = module(*data)
                loss_value = loss[0](*prediction, *label)
                loss_record.append(loss_value.detach())
                loss_value.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                module.train(False)
            # ----- 计算校验集的loss和metric
            loss_value = evaluate(module, valid_data_loader, loss, ncols)
            metric_value = loss_value[metric_loss_id]
            if best_metric_value is None or (metric_value != best_metric_value
                                             and is_higher_better == (metric_value > best_metric_value)):
                best_state_dict = deepcopy(module.state_dict())
                best_step = step
                best_metric_value = metric_value
            with torch.no_grad():
                print('train {}: {}; valid '.format(
                    type(loss[0]).__name__, torch.tensor(loss_record[-verbose:],
                                                         device=loss_record[-1].device).mean()), end='')
            for a, b in zip(loss_value, loss):
                print('{}: {}, '.format(type(b).__name__, a), end='')
            print()
            # ------ 提前停止的策略
            if step - best_step >= early_stopping_rounds > 0:
                break
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    finally:
        module.load_state_dict(best_state_dict)
        module.train(False)
    return best_metric_value, torch.tensor(loss_record, device=loss_record[0].device)


class NumericEmbedding(torch.nn.Module):
    """
    参考torch.nn.Embedding文档，将数值型特征也变成相同的形状。
    实际上就是将输入的张量扩展一个为1的维度之后，加上一个没有常数项的全连接层（和一个非线性激活层）
    """
    def __init__(self, input_dim: List[int], emb_dim):
        super(NumericEmbedding, self).__init__()
        size = [1] * (len(input_dim) + 1) + [emb_dim]
        size[-2] = input_dim[-1]
        self.weight = torch.nn.Parameter(torch.empty(size))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = torch.unsqueeze(inputs, -1)
        output = output * self.weight
        return output
