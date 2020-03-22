# 尽量使用tensor类型传递数据，不保证numpy类型能正确运行
from copy import deepcopy
from inspect import isbuiltin, isfunction
from functools import wraps
from typing import List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score as _roc_auc_score
from tqdm import tqdm as _tqdm


def round_wrap(number, ndigits=None):
    if ndigits is None:
        return number
    else:
        return round(number, ndigits)


N_COLS = 80
@wraps(_tqdm)
def tqdm(*args, **kwargs):
    with _tqdm(*args, ncols=N_COLS, **kwargs) as t:
        try:
            for _ in t:
                yield _
        except KeyboardInterrupt:
            t.close()
            raise KeyboardInterrupt


def roc_auc_score(prediction, target):
    return _roc_auc_score(target.cpu(), prediction.cpu())


def get_name(x):
    if isbuiltin(x) or isfunction(x):
        return x.__name__
    else:
        return x.__class__.__name__


def stack(x):
    if isinstance(x[0], (list, tuple)):
        outputs = [[] for _ in x[0]]
        for _ in x:
            for i in range(len(_)):
                item = _[i]
                outputs[i].append(item)
        outputs = [stack(_) for _ in outputs]
        return outputs
    else:
        return torch.cat(x)


def to(x, *args, **kwargs):
    if isinstance(x, (list, tuple, np.ndarray)):
        return tuple(to(_, *args, **kwargs) for _ in x)
    else:
        return x.to(*args, **kwargs)


def _evaluate_per_batch(module: torch.nn.Module, data_loader, criterion=None, metric=(), verbose=False, ndigits=None,
                        name=None):
    """

    :param module:
    :param data_loader:
    :param criterion:
    :param metric: 单个metric，或者list of metric
    :param verbose:
    :param ndigits:
    :param name:
    :return:
    """
    if not isinstance(metric, (list, tuple)):
        metric = [metric]
    loss_value = []
    metrics_value = [[] for _ in metric]
    with torch.no_grad():
        # -------- 获得target ----------
        module.train(False)

        for batch_data, batch_target in tqdm(data_loader, disable=not verbose):
            batch_prediction = module(*batch_data)
            if criterion is not None:
                loss_value.append(criterion(*batch_prediction, batch_data).item())
            for value, m in zip(metrics_value, metric):
                value.append(m(*batch_prediction, batch_target).item())

        # -------- 计算损失 ---------
        if criterion is not None:
            loss_value = np.mean(loss_value)
        if criterion is not None and verbose:
            print('{} - {}: {}'.format(name, get_name(criterion), round_wrap(loss_value, ndigits)), end='')

        # -------- 计算metric
        for i in range(len(metrics_value)):
            metrics_value[i] = np.mean(metrics_value[i])
        if verbose:
            for v, m in zip(metrics_value, metric):
                print(' - {}: {}'.format(get_name(m), round_wrap(v, ndigits)), end='')
        if verbose:
            print()
    if len(metrics_value) == 0:
        return loss_value, None
    elif len(metrics_value) == 1:
        return loss_value, metrics_value[0]
    else:
        return loss_value, metrics_value


def evaluate(module: torch.nn.Module, data_loader, criterion=None, metric=(), verbose=False, ndigits=None, name=None,
             evaluate_per_batch=False):
    """

    :param module:
    :param data_loader:
    :param criterion:
    :param metric: 单个metric，或者list of metric
    :param verbose:
    :param ndigits:
    :param name:
    :param evaluate_per_batch: 如果True，每个batch计算一次，最后在取平均，能减少存储计算结果，减少显存占用
    :return:
    """
    if evaluate_per_batch:
        return _evaluate_per_batch(module, data_loader, criterion, metric, verbose, ndigits, name)

    if not isinstance(metric, (list, tuple)):
        metric = [metric]
    loss_value = None
    metrics_value = []
    with torch.no_grad():
        # -------- 获得target ----------
        module.train(False)
        target = []
        prediction = []
        for batch_data, batch_target in tqdm(data_loader, disable=not verbose):
            target.append(batch_target)
            batch_prediction = module(*batch_data)
            prediction.append(batch_prediction)
        target = stack(target)
        prediction = stack(prediction)

        # -------- 计算损失 ---------
        if criterion is not None:
            loss_value = criterion(*prediction, target).item()
            if verbose:
                print('{} - {}: {}'.format(name, get_name(criterion), round_wrap(loss_value, ndigits)), end='')

        # -------- 计算metric
        for _ in metric:
            metric_value = _(*prediction, target).item()
            if verbose:
                print(' - {}: {}'.format(get_name(_), round_wrap(metric_value, ndigits)), end='')
            metrics_value.append(metric_value)
        if verbose:
            print()
    if len(metrics_value) == 0:
        return loss_value, None
    elif len(metrics_value) == 1:
        return loss_value, metrics_value[0]
    else:
        return loss_value, metrics_value


def _fit_per_batch(module: torch.nn.Module, data_loader, criterion, optimizer, scheduler=None, epochs=1, metrics=(),
                   is_higher_better=None, valid_data_loader=None, early_stopping_rounds=None, ndigits=None,
                   verbose=False, eval_value=None) -> (float, List[str]):
    # TODO 目前只兼容module输出多个张量的情况，但是还不兼容有多个target的情况
    """

    :param module: pytorch Module
    :param data_loader: pytorch DataLoader，训练集
    :param criterion: 训练目标损失的类，必须符合pytorch的规范
    :param optimizer: 优化器
    :param scheduler: 用于动态调节学习率
    :param epochs: 最大训练轮数
    :param metrics: 准则函数，List，必须符合pytorch的规范
    :param is_higher_better: 准则是否越大越好
    :param valid_data_loader: 校验集
    :param early_stopping_rounds: 当metric在一定epoch之后还不提高，就提前停止训练
    :param ndigits: 保留小数点后
    :param verbose: 是否打印训练进度
    :param eval_value:
    :return: metric value or loss value of valid
    """
    if len(metrics) > 0:
        assert is_higher_better is not None, 'parameter is_higher_better is missing'

    # 状态变量
    best_state_dict = deepcopy(module.state_dict())
    best_epoch = -1
    best_eval_value = eval_value
    try:
        for epoch in range(1, epochs + 1):
            if verbose:
                print('epoch {}/{}'.format(epoch, epochs))
            # --------- 训练参数 ------------
            module.train(True)
            losses = []
            metrics_value = [[] for _ in metrics]
            for batch_data, batch_target in tqdm(data_loader, disable=not verbose):
                optimizer.zero_grad()
                batch_prediction = module(*batch_data)
                loss = criterion(*batch_prediction, batch_target)
                losses.append(loss.item())

                if hasattr(module, 'l1_l2_penalty'):
                    (loss + module.l1_l2_penalty()).backward()
                else:
                    loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                for v, m in zip(metrics_value, metrics):
                    v.append(m(*batch_prediction, batch_target).item())
            module.train(False)

            if verbose:
                print('train - {}: {}'.format(get_name(criterion), round_wrap(np.mean(losses), ndigits)), end='')
            # ---------计算训练集的metric ----------
                for v, m in zip(metrics_value, metrics):
                    print(' - {}: {}'.format(get_name(m), round_wrap(np.mean(v), ndigits)), end='')
                print()

            # ----- 计算校验集的loss和metric
            if valid_data_loader is not None:
                # 计算校验集的metric
                loss_value, metrics_value = _evaluate_per_batch(
                    module, valid_data_loader, criterion, metrics, verbose, ndigits, 'valid')
                if not isinstance(metrics_value, list):
                    metrics_value = [metrics_value]

                # 使用第一个metric作为early stopping的依据
                if len(metrics_value) > 0:
                    metric_value = metrics_value[0]
                else:
                    metric_value = loss_value

                if best_eval_value is None or (metric_value != best_eval_value
                                               and is_higher_better == (metric_value > best_eval_value)):
                    best_state_dict = deepcopy(module.state_dict())
                    best_epoch = epoch
                    best_eval_value = metric_value

                # 提前停止的策略
                if early_stopping_rounds is not None and epoch - best_epoch >= early_stopping_rounds:
                    break
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    finally:
        module.load_state_dict(best_state_dict)
        module.train(False)
    return best_eval_value


def fit(module: torch.nn.Module, data_loader, criterion, optimizer, scheduler=None, epochs=1, metrics=(),
        is_higher_better=None, valid_data_loader=None, early_stopping_rounds=None, ndigits=None, verbose=False,
        eval_value=None, evaluate_per_batch=False) -> (float, List[str]):
    # TODO 目前只兼容module输出多个张量的情况，但是还不兼容有多个target的情况
    """

    :param module: pytorch Module
    :param data_loader: pytorch DataLoader，训练集
    :param criterion: 训练目标损失的类，必须符合pytorch的规范
    :param optimizer: 优化器
    :param scheduler: 用于动态调节学习率
    :param epochs: 最大训练轮数
    :param metrics: 准则函数，List，必须符合pytorch的规范
    :param is_higher_better: 准则是否越大越好
    :param valid_data_loader: 校验集
    :param early_stopping_rounds: 当metric在一定epoch之后还不提高，就提前停止训练
    :param ndigits: 保留小数点后
    :param verbose: 是否打印训练进度
    :param eval_value:
    :param evaluate_per_batch: 如果True，每个batch计算一次，最后在取平均，能减少存储计算结果，减少显存占用
    :return: metric value or loss value of valid
    """
    if evaluate_per_batch:
        return _fit_per_batch(module, data_loader, criterion, optimizer, scheduler, epochs, metrics, is_higher_better,
                              valid_data_loader, early_stopping_rounds, ndigits, verbose, eval_value)

    if len(metrics) > 0:
        assert is_higher_better is not None, 'parameter is_higher_better is missing'

    # 状态变量
    best_state_dict = deepcopy(module.state_dict())
    best_epoch = -1
    best_eval_value = eval_value
    try:
        for epoch in range(1, epochs + 1):
            if verbose:
                print('epoch {}/{}'.format(epoch, epochs))
            # --------- 训练参数 ------------
            module.train(True)
            target = []
            prediction = []
            for batch_data, batch_target in tqdm(data_loader, disable=not verbose):
                optimizer.zero_grad()
                batch_prediction = module(*batch_data)

                if isinstance(batch_prediction, (list, tuple)):
                    loss = criterion(*batch_prediction, batch_target)
                    prediction.append(tuple(_.detach() for _ in batch_prediction))
                else:
                    loss = criterion(batch_prediction, batch_target)
                    prediction.append(batch_prediction.detach())
                target.append(batch_target.detach())

                if hasattr(module, 'l1_l2_penalty'):
                    (loss + module.l1_l2_penalty()).backward()
                else:
                    loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            module.train(False)

            with torch.no_grad():
                # ---------- 计算训练集的loss --------
                target = stack(target)
                prediction = stack(prediction)
                if isinstance(prediction, (list, tuple)):
                    loss_value = criterion(*prediction, target).item()
                else:
                    loss_value = criterion(prediction, target).item()
                if verbose:
                    print('train - {}: {}'.format(get_name(criterion), round_wrap(loss_value, ndigits)), end='')

                # ---------计算训练集的metric ----------
                if len(metrics) > 0:
                    for metric in metrics:
                        if isinstance(prediction, (list, tuple)):
                            metric_value = metric(*prediction, target).item()
                        else:
                            metric_value = metric(prediction, target).item()
                        if verbose:
                            print(' - {}: {}'.format(get_name(metric), round_wrap(metric_value, ndigits)), end='')
                if verbose:
                    print()

            # ----- 计算校验集的loss和metric
            if valid_data_loader is not None:
                # 计算校验集的metric
                loss_value, metrics_value = evaluate(
                    module, valid_data_loader, criterion, metrics, verbose, ndigits, 'valid', evaluate_per_batch)
                if not isinstance(metrics_value, list):
                    metrics_value = [metrics_value]

                # 使用第一个metric作为early stopping的依据
                if len(metrics_value) > 0:
                    metric_value = metrics_value[0]
                else:
                    metric_value = loss_value

                if best_eval_value is None or (metric_value != best_eval_value
                                               and is_higher_better == (metric_value > best_eval_value)):
                    best_state_dict = deepcopy(module.state_dict())
                    best_epoch = epoch
                    best_eval_value = metric_value

                # 提前停止的策略
                if early_stopping_rounds is not None and epoch - best_epoch >= early_stopping_rounds:
                    break
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    finally:
        module.load_state_dict(best_state_dict)
        module.train(False)
    return best_eval_value
