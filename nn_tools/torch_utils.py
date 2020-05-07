import math
import pickle
import time
import traceback
from copy import deepcopy
from typing import List

import torch

from .utils import tqdm


def forever_iter(iterable):
    while True:
        for _ in iterable:
            yield _


def try_forward(module: torch.nn.Module, data):
    # catch module.forward的错误，如果正常运行，返回module(*data)的结果，否则返回None
    try:
        pred = module(*data)
        return pred
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except Exception as exception:
        error_info = traceback.format_exc()
        if 'CUDA out of memory' in error_info:
            raise exception
        
        print('an error occurs!')
        timestamp = time.time()
        with open('error_batch_data_{}.pkl'.format(timestamp), 'bw') as file:
            pickle.dump(data, file)
        with open('error_batch_traceback_{}.txt'.format(timestamp), 'w') as file:
            file.write(error_info)
        return None


def evaluate(module: torch.nn.Module, data, metrics: list) -> torch.Tensor:
    with torch.no_grad():
        module.eval()
        loss_value = [[] for _ in metrics]
        for data, label in tqdm(data, ascii=True):
            prediction = try_forward(module, data)
            if prediction is None:
                continue
            for a, b in zip(loss_value, metrics):
                a.append(b(*prediction, *label))
        loss_value = [torch.tensor([x for x in array if x is not None], device=data[0].device).mean()
                      for array in loss_value]
    return torch.tensor(loss_value, device=loss_value[0].device)


def fit(module: torch.nn.Module, train_data, valid_data, optimizer, max_step, loss, metrics: list, is_higher_better,
        evaluate_per_steps=None, early_stopping=-1, scheduler=None, init_metric_value=None, evaluate_fn=evaluate):
    # 状态变量
    print('using {} as training loss, using {}({} is better) as early stopping metric'.format(
        type(loss).__name__, type(metrics[0]).__name__, 'higher' if is_higher_better else 'lower'))
    evaluate_per_steps = evaluate_per_steps or max_step

    best_state_dict = deepcopy(module.state_dict())
    best_step = -1
    best_metric_value = init_metric_value
    loss_record = []
    step = 0
    generator = forever_iter(train_data)
    try:
        while step < max_step:
            time.sleep(0.5)
            module.train(True)
            for _ in tqdm(range(evaluate_per_steps), ascii=True):
                step += 1
                # --------- 训练参数 ------------
                data, label = next(generator)
                optimizer.zero_grad()
                prediction = try_forward(module, data)
                if prediction is None:
                    continue
                loss_value = loss(*prediction, *label)
                loss_record.append(loss_value.detach())
                if loss_value is not None:
                    device = loss_value.device
                    loss_value.backward()
                    optimizer.step()
                if scheduler:
                    scheduler.step()
            # ----- 计算校验集的loss和metric
            metrics_values = evaluate_fn(module, valid_data, metrics)
            init_metric_value = metrics_values[0]
            if best_metric_value is None or (init_metric_value != best_metric_value
                                             and is_higher_better == (init_metric_value > best_metric_value)):
                best_state_dict = deepcopy(module.state_dict())
                best_step = step
                best_metric_value = init_metric_value
                torch.save(module, '{}.checkpoint'.format(step))
            with torch.no_grad():
                print('step {} train {}: {}; valid '.format(
                    step, type(loss).__name__, torch.tensor(
                        [x for x in loss_record[-evaluate_per_steps:] if x is not None]).mean()), end='')
            for a, b in zip(metrics_values, metrics):
                print('{}: {}, '.format(type(b).__name__, a), end='')
            print()
            # ------ 提前停止的策略
            if step - best_step >= early_stopping > 0:
                break
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    finally:
        module.load_state_dict(best_state_dict)
    return best_metric_value, loss_record


class NumericEmbedding(torch.nn.Module):
    """
    参考torch.nn.Embedding文档，将数值型特征也变成相同的形状。
    实际上就是将输入的张量扩展一个为1的维度之后，加上一个没有常数项的全连接层
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
