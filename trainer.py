import argparse
from time import time
import numpy as np
# from model import my_DRMM
# from data_load import Interactions, my_Datasets, Dataloaders
# from evaluation import MeanReciprocalRank, NormalizedDiscountedCumulativeGain
import os
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
import time
import typing
import pandas as pd
import pickle

class RankCrossEntropyLoss(nn.Module):
    """Creates a criterion that measures rank cross entropy loss."""

    __constants__ = ['num_neg']

    def __init__(self, num_neg: int = 1):
        """
        :class:`RankCrossEntropyLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        """
        super().__init__()
        self.num_neg = num_neg

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, stage='train'):
        """
        Calculate rank cross entropy loss.

        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Rank cross loss.
        """
        if stage == 'train':
            logits = y_pred[::(self.num_neg + 1), :]
            labels = y_true[::(self.num_neg + 1), :]
            for neg_idx in range(self.num_neg):
                neg_logits = y_pred[(neg_idx + 1)::(self.num_neg + 1), :]
                neg_labels = y_true[(neg_idx + 1)::(self.num_neg + 1), :]
                logits = torch.cat((logits, neg_logits), dim=-1)
                labels = torch.cat((labels, neg_labels), dim=-1)
            return -torch.mean(
                torch.sum(
                    labels * torch.log(F.softmax(logits, dim=-1) + torch.finfo(float).eps),
                    dim=-1
                )
            )
        elif stage=='dev':
            logits = torch.reshape(y_pred, (1, -1))
            labels = torch.reshape(y_true, (1, -1))
            return -torch.mean(
                torch.sum(
                    labels * torch.log(F.softmax(logits, dim=-1) + torch.finfo(float).eps),
                    dim=-1
                )
            )


    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        """`num_neg` setter."""
        self._num_neg = value

class EarlyStopping:
    """
    EarlyStopping stops training if no improvement after a given patience.

    :param patience: Number fo events to wait if no improvement and then
        stop the training.
    :param should_decrease: The way to judge the best so far.
    :param key: Key of metric to be compared.
    """

    def __init__(
        self,
        patience: typing.Optional[int] = None,
        should_decrease: bool = None,
        key: typing.Any = None
    ):
        """Early stopping Constructor."""
        self._patience = patience
        self._key = key
        self._best_so_far = 0
        self._epochs_with_no_improvement = 0
        self._is_best_so_far = False
        self._early_stop = False

    def state_dict(self) -> typing.Dict[str, typing.Any]:
        """A `Trainer` can use this to serialize the state."""
        return {
            'patience': self._patience,
            'best_so_far': self._best_so_far,
            'is_best_so_far': self._is_best_so_far,
            'epochs_with_no_improvement': self._epochs_with_no_improvement,
        }

    def load_state_dict(
        self,
        state_dict: typing.Dict[str, typing.Any]
    ) -> None:
        """Hydrate a early stopping from a serialized state."""
        self._patience = state_dict["patience"]
        self._is_best_so_far = state_dict["is_best_so_far"]
        self._best_so_far = state_dict["best_so_far"]
        self._epochs_with_no_improvement = \
            state_dict["epochs_with_no_improvement"]

    def update(self, result: list):
        """Call function."""
        score = result[self._key]
        if score > self._best_so_far:
            self._best_so_far = score
            self._is_best_so_far = True
            self._epochs_with_no_improvement = 0
        else:
            self._is_best_so_far = False
            self._epochs_with_no_improvement += 1

    @property
    def best_so_far(self) -> bool:
        """Returns best so far."""
        return self._best_so_far

    @property
    def is_best_so_far(self) -> bool:
        """Returns true if it is the best so far."""
        return self._is_best_so_far

    @property
    def should_stop_early(self) -> bool:
        """Returns true if improvement has stopped for long enough."""
        if not self._patience:
            return False
        else:
            return self._epochs_with_no_improvement >= self._patience


"""Timer."""

import time


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        """Timer constructor."""
        self.reset()

    def reset(self):
        """Reset timer."""
        self.running = True
        self.total = 0
        self.start = time.time()

    def resume(self):
        """Resume."""
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        """Stop."""
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    @property
    def time(self):
        """Return time."""
        if self.running:
            return self.total + time.time() - self.start
        return self.total




class AverageMeter(object):
    """
    Computes and stores the average and current value.

    Examples:
        >>> am = AverageMeter()
        >>> am.update(1)
        >>> am.avg
        1.0
        >>> am.update(val=2.5, n=2)
        >>> am.avg
        2.0

    """

    def __init__(self):
        """Average meter constructor."""
        self.reset()

    def reset(self):
        """Reset AverageMeter."""
        self._val = 0.
        self._avg = 0.
        self._sum = 0.
        self._count = 0.

    def update(self, val, n=1):
        """Update value."""
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count

    @property
    def avg(self):
        """Get avg."""
        return self._avg


class Trainer:
    """
    MatchZoo tranier.

    :param model: A :class:`BaseModel` instance.
    :param optimizer: A :class:`optim.Optimizer` instance.
    :param trainloader: A :class`DataLoader` instance. The dataloader
        is used for training the model.
    :param validloader: A :class`DataLoader` instance. The dataloader
        is used for validating the model.
    :param device: The desired device of returned tensor. Default:
        if None, use the current device. If `torch.device` or int,
        use device specified by user. If list, use data parallel.
    :param start_epoch: Int. Number of starting epoch.
    :param epochs: The maximum number of epochs for training.
        Defaults to 10.
    :param validate_interval: Int. Interval of validation.
    :param scheduler: LR scheduler used to adjust the learning rate
        based on the number of epochs.
    :param clip_norm: Max norm of the gradients to be clipped.
    :param patience: Number fo events to wait if no improvement and
        then stop the training.
    :param key: Key of metric to be compared.
    :param checkpoint: A checkpoint from which to continue training.
        If None, training starts from scratch. Defaults to None.
        Should be a file-like object (has to implement read, readline,
        tell, and seek), or a string containing a file name.
    :param save_dir: Directory to save trainer.
    :param save_all: Bool. If True, save `Trainer` instance; If False,
        only save model. Defaults to False.
    :param verbose: 0, 1, or 2. Verbosity mode. 0 = silent,
        1 = verbose, 2 = one log line per epoch.
    """

    def __init__(
        self,
        model,
        optimizer,
        
        validloader,
        num_neg,
        metrics,
        trainloader = None,   #just for test
        device = None,
        start_epoch: int = 1,
        epochs: int = 10,
        validate_interval = None,
        scheduler = None,
        clip_norm = None,
        patience = None,
        key = None,
        checkpoint = None,
        save_dir= None,
        save_all = False,
        verbose = 1,
        constant=None,
        **kwargs
    ):
        """Base Trainer constructor."""
        self._load_model(model, device)
        self._load_dataloader(
            trainloader, validloader, validate_interval
        )

        self._optimizer = optimizer
        self._scheduler = scheduler
        self._clip_norm = clip_norm
        self._criterions = [RankCrossEntropyLoss(num_neg=num_neg)]
        self._constant = constant
        self._metrics = metrics
        if not key:
            key = self._metrics[0]
        self._early_stopping = EarlyStopping(
            patience=patience,
            key=key
        )

        self._start_epoch = start_epoch
        self._epochs = epochs
        self._iteration = 0
        self._verbose = verbose
        self._save_all = save_all

        self._load_path(checkpoint, save_dir)

    def _load_dataloader(
        self,
        trainloader,
        validloader,
        validate_interval = None
    ):
        """
        Load trainloader and determine validate interval.

        :param trainloader: A :class`DataLoader` instance. The dataloader
            is used to train the model.
        :param validloader: A :class`DataLoader` instance. The dataloader
            is used to validate the model.
        :param validate_interval: int. Interval of validation.
        """
        self._trainloader = trainloader
        self._validloader = validloader
        
        self._validate_interval = validate_interval

    def _load_model(
        self,
        model,
        device = None
    ):
        """
        Load model.

        :param model: :class:`BaseModel` instance.
        :param device: The desired device of returned tensor. Default:
            if None, use the current device. If `torch.device` or int,
            use device specified by user. If list, use data parallel.
        """

        # self._task = model.params['task']
        # self._data_parallel = False
        self._model = model

        if isinstance(device, list) and len(device):
            self._data_parallel = True
            self._model = torch.nn.DataParallel(self._model, device_ids=device)
            self._device = device[0]
        else:
            self._device = device

        self._model.to(self._device)

    def _load_path(
        self,
        checkpoint: typing.Union[str, Path],
        save_dir: typing.Union[str, Path],
    ):
        """
        Load save_dir and Restore from checkpoint.

        :param checkpoint: A checkpoint from which to continue training.
            If None, training starts from scratch. Defaults to None.
            Should be a file-like object (has to implement read, readline,
            tell, and seek), or a string containing a file name.
        :param save_dir: Directory to save trainer.

        """
        if not save_dir:
            save_dir = Path('.').joinpath('save')
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)

        self._save_dir = Path(save_dir)
        # Restore from checkpoint

        if checkpoint:
            if self._save_all:
                self.restore(checkpoint)
            else:
                self.restore_model(checkpoint)

    def _backward(self, loss):
        """
        Computes the gradient of current `loss` graph leaves.

        :param loss: Tensor. Loss of model.

        """
        self._optimizer.zero_grad()
        loss.backward()
        if self._clip_norm:
            nn.utils.clip_grad_norm_(
                self._model.parameters(), self._clip_norm
            )
        self._optimizer.step()

    def _run_scheduler(self):
        """Run scheduler."""
        if self._scheduler:
            self._scheduler.step()


    def run_test(self):

        if self._verbose:
            self._epoch = 0
            result = self.evaluate(self._validloader)
            if self._constant.logger is not None:
                self._constant.logger.info('  Validation: ' + ' - '.join(
                    f'{k}: {round(v, 4)}' for k, v in result.items()))
                
        pickle.dump(self.scores_df, open(self._save_dir.joinpath('score_df.pkl'), 'wb'))


    def evaluate(
        self,
        dataloader,
    ):
        """
        Evaluate the model.

        :param dataloader: A DataLoader object to iterate over the data.

        """
        # result = dict()
        metrics = self.predict(dataloader)
        # y_true = dataloader.label
        # id_left = dataloader.id_left
        #
        # if isinstance(self._task, tasks.Classification):
        #     for metric in self._task.metrics:
        #         result[metric] = metric(y_true, y_pred)
        # else:
        #     for metric in self._task.metrics:
        #         result[metric] = self._eval_metric_on_data_frame(
        #             metric, id_left, y_true, y_pred.squeeze(axis=-1))
        # return result
        return metrics

    @classmethod
    def _eval_metric_on_data_frame(
        cls,
        metric,
        record_id,
        y_true,
        y_pred
    ):
        """
        Eval metric on data frame.

        This function is used to eval metrics for `Ranking` task.

        :param metric: Metric for `Ranking` task.
        :param id_left: id of input left. Samples with same id_left should
            be grouped for evaluation.
        :param y_true: Labels of dataset.
        :param y_pred: Outputs of model.
        :return: Evaluation result.

        """
        eval_df = pd.DataFrame(data={
            'id': record_id,
            'true': y_true,
            'pred': y_pred
        })
        val = eval_df.groupby(by='id').apply(
            lambda df: metric(df['true'].values, df['pred'].values)
        ).mean()
        return val

    def predict(
        self,
        dataloader
    ) -> np.array:
        """
        Generate output predictions for the input samples.

        :param dataloader: input DataLoader
        :return: predictions

        """
        self.results_score_list = []
        timer_all = []
        with torch.no_grad():
            self._model.eval()
            logs = []
            valid_loss_a = AverageMeter()
            for batch in dataloader:
                inputs = batch[0]
                # for k in inputs.keys():
                #     inputs[k] = inputs[k].to(device)
                keys = list(inputs.keys())
                num_ = inputs[keys[0]].size()[0]
                block_num = 20
                num_split = np.array_split(list(range(num_)), block_num)
                y_pred_all = []
                x_pred_all = []
                t1 = time.time()
                for i in range(block_num):
                    inputs_ = {}
                    for k in keys:
                        inputs_[k] = inputs[k][num_split[i]]

                    # if self._epoch <= 5:
                    #     y_pred = self._model.forward1(inputs_)
                    # else:
                    #     y_pred = self._model.forward(inputs_)
                    y_pred = self._model.forward(inputs_)
                
                    y_pred_all.append(y_pred)
                y_pred_all = torch.cat(y_pred_all, dim=0)
                timer_all.append(time.time()-t1) 
                y_true = batch[1]
                valid_loss = torch.sum(
                    *[c(y_pred_all, y_true, stage='dev') for c in self._criterions]
                )
                valid_loss_a.update(valid_loss.item())
                y_pred_all = y_pred_all.detach().cpu().numpy()
                y_true = y_true.detach().cpu().numpy()
                record_id = inputs['record_id'].cpu().numpy()
                candidates = inputs['id_right'].cpu().numpy()
                left_ids = inputs['id_left'].cpu().numpy()
                score_df = pd.DataFrame(data={
                    'record_id': record_id,
                    'id_left': left_ids,
                    'id_right': candidates,
                    'label': y_true.squeeze(axis=-1),
                    'preds': y_pred_all.squeeze(axis=-1)
                })
                score_df_group = score_df.groupby(by='record_id')
                for sg in score_df_group:
                    self.score_fun(sg[1]['record_id'].values, sg[1]['id_left'].values, sg[1]['id_right'].values, sg[1]['label'].values, sg[1]['preds'].values)
            self._constant.logger.info(f'Valid Loss-{valid_loss_a.avg:.3f}]:')
            self._model.train()
            self._constant.logger.info(f'cost time for one query:{np.mean(timer_all)}, sum for all queries: {np.sum(timer_all)}')
            self.scores_df = pd.DataFrame(self.results_score_list,
                                          columns=['record_id', 'id_left', 'id_right', 'labels', 'scores_ie'])
            
            metrics_results = self.metric_cal(self.scores_df)
            return metrics_results

    def metric_cal(self, scores_df):
        logs = []
        for idx in range(len(scores_df)):
            y_true = scores_df.iloc[idx]['labels']
            y_pre = scores_df.iloc[idx]['scores_ie']
            result = dict()
            for metric in self._metrics:
                val = metric(y_true, y_pre)
                result[metric] = val
            logs.append(result)
        metrics_results = {}
        for metric in logs[0].keys():
            metrics_results[metric] = sum([log[metric] for log in logs]) / len(logs)
        return metrics_results

    def score_fun(self, record_ids, id_lefts, id_rights, labels, preds):
        sort_idx = np.argsort(id_rights)
        self.results_score_list.append(
            [record_ids[0], id_lefts[0], id_rights[0], labels[sort_idx], preds[sort_idx]])


    def _save(self):
        """Save."""
        if self._save_all:
            self.save()
        else:
            self.save_model()

    def save_model(self):
        """Save the model."""
        checkpoint = self._save_dir.joinpath('model.pt')
        
        torch.save(self._model.state_dict(), checkpoint)

    def save(self):
        """
        Save the trainer.

        `Trainer` parameters like epoch, best_so_far, model, optimizer
        and early_stopping will be savad to specific file path.

        :param path: Path to save trainer.

        """
        checkpoint = self._save_dir.joinpath('trainer.pt')
        
        model = self._model.state_dict()
        state = {
            'epoch': self._epoch,
            'model': model,
            'optimizer': self._optimizer.state_dict(),
            'early_stopping': self._early_stopping.state_dict(),
        }
        if self._scheduler:
            state['scheduler'] = self._scheduler.state_dict()
        torch.save(state, checkpoint)

    def restore_model(self, checkpoint):
        """
        Restore model.

        :param checkpoint: A checkpoint from which to continue training.

        """
        state = torch.load(checkpoint, map_location=self._device)

        self._model.load_state_dict(state)

    def restore(self, checkpoint = None):
        """
        Restore trainer.

        :param checkpoint: A checkpoint from which to continue training.

        """
        state = torch.load(checkpoint, map_location=self._device)
        if self._data_parallel:
            self._model.module.load_state_dict(state['model'])
        else:
            self._model.load_state_dict(state['model'])
        self._optimizer.load_state_dict(state['optimizer'])
        self._start_epoch = state['epoch'] + 1
        self._early_stopping.load_state_dict(state['early_stopping'])
        if self._scheduler:
            self._scheduler.load_state_dict(state['scheduler'])