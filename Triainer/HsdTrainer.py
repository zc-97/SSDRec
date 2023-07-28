from recbole.trainer import Trainer
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
from recbole.data.interaction import Interaction
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.evaluator import Evaluator, Collector
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str, \
    EvaluatorType, KGDataLoaderState, get_tensorboard, set_color, get_gpu_usage
from time import time


class HsdTrainer(Trainer):

    def __init__(self, config, model):
        super(HsdTrainer, self).__init__(config, model)

        # 加入scheduler
        if self.config['scheduler']:
            lr_dc_step = self.config['step_size']
            lr_dc = self.config['gamma']
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_dc_step, gamma=lr_dc)
        else:
            self.scheduler = False

        self.global_train_batches = 0
        self.tau = self.config['gumbel_temperature']
        self.gumbel_tau_anneal = self.config['is_gumbel_tau_anneal']

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if self.config['reg_weight'] and weight_decay and weight_decay * self.config['reg_weight'] > 0:
            self.logger.warning(
                'The parameters [weight_decay] and [reg_weight] are specified simultaneously, '
                'which may lead to double regularization.'
            )

        if learner.lower() == 'adam':
            if self.config['model'] == 'fmlp':
                optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
            elif 'HSD' in self.config['model']:
                optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
            else:
                optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            if weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    # def step_lr

    def drop_rate_schedule_curriculum(self,iteration):
        # drop_rate = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

        drop_rate = np.linspace(0.2 ** 1, 0, 10)
        if iteration < 10:
            return drop_rate[iteration]
            # return 0.0
        else:
            return 0.0

    def _gumbel_softmax_tempreture_anneal(self):
        r = 1e-3
        self.tau = max(1e-3, self.tau * np.exp(- r * self.global_train_batches))

    def drop_rate_schedule(self,iteration):
        # drop_rate = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

        drop_rate = np.linspace(0, 0.2, 4)
        if iteration < 4:
            return 0.0
            # return 0.0
        else:
            return 0.0

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.
            show_progress (bool): Show the progress of training epoch. Defaults to ``False``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, it will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()

        '在这判断使用课程学习loss还是正常loss'
        if 'HSD' not in str(self.model):
            loss_func = self.model.calculate_loss
        else:
            loss_func = self.model.calculate_loss_curriculum


        # loss_func = loss_func or self.model.calculate_loss_curriculum
        total_loss = None
        total_rec = 0
        total_gen = 0
        total_percent = 0
        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=200,
                desc=set_color(f"Train {epoch_idx:>5}", 'pink'),
            ) if show_progress else train_data
        )
        for batch_idx, interaction in enumerate(iter_data):
            # 这里记录global batches，为了对gumbel softmax做退火
            self.global_train_batches += 1
            # 每100个batch退火一次
            if self.global_train_batches % 40 == 0 and self.gumbel_tau_anneal:
                self._gumbel_softmax_tempreture_anneal()
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()

            clean_seq_percent = 100
            L_rec = torch.tensor(0)
            generated_seq_loss = torch.tensor(0)
            if 'HSD' in str(self.model):
                losses, clean_seq_percent, L_rec, generated_seq_loss = loss_func(interaction, self.drop_rate_schedule_curriculum(epoch_idx), self.tau)  # reconstruction_loss [batch_size]
            else:
                losses = loss_func(interaction)

            total_percent += clean_seq_percent.__float__()

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            total_rec = total_rec + L_rec.item()
            total_gen = total_gen + generated_seq_loss.item()
            self._check_nan(loss)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('gumbel tau: ' + '%.2f' % self.tau, 'yellow') + ', ' +
                                          set_color('clean_seq: ', 'blue') + '%.2f' % clean_seq_percent.__float__() + '%'+ ', ' +
                                          set_color('原始loss: ', 'blue') + '%.6f' % loss.data.__float__() + ', ' +
                                          set_color('重建loss: ', 'blue') + '%.6f' % L_rec.data.__float__() + ', ' +
                                          set_color('推荐损失: ', 'blue') + '%.6f' % generated_seq_loss.data.__float__() +''
                                          )

        # 加入scheduler
        if self.scheduler:
            self.logger.info(set_color('Successfully utilize lr_scheduler strategy','pink'))
            self.scheduler.step()
        return total_loss, total_percent / len(train_data), total_rec, total_gen

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
            callback_fn (callable): Optional callback function executed at end of epoch.
                                    Includes (epoch_idx, valid_score) input arguments.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if saved and self.start_epoch >= self.epochs:
            self._save_checkpoint(-1, verbose=verbose)

        self.eval_collector.data_collect(train_data)
        if self.config['train_neg_sample_args'].get('dynamic', 'none') != 'none':
            train_data.get_model(self.model)
        valid_step = 0
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            train_loss, clean_item_percent, total_rec, total_gen = self._train_epoch(train_data, epoch_idx, show_progress=show_progress)
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info(train_loss_output)
            self._add_train_loss_to_tensorboard(epoch_idx, train_loss)
            self.wandblogger.log_metrics({'epoch': epoch_idx, 'train_loss': train_loss, 'train_step':epoch_idx}, head='train')

            self.logger.info(set_color('clean item percent', 'yellow') + ': %.2f' % clean_item_percent + '%')
            self.logger.info(set_color('推荐loss', 'blue') + ': %.4f' % total_gen)
            self.logger.info(set_color('重建', 'blue') + ': %.4f' % total_rec )
            # eval
            if self.eval_step <= 0 or not valid_data:
                if saved:
                    self._save_checkpoint(epoch_idx, verbose=verbose)
                continue
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score,
                    self.best_valid_score,
                    self.cur_step,
                    max_step=self.stopping_step,
                    bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = (set_color("epoch %d evaluating", 'green') + " [" + set_color("time", 'blue')
                                    + ": %.2fs, " + set_color("valid_score", 'blue') + ": %f, " + set_color("cur_step", 'yellow') + ": %d]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score, self.cur_step)
                valid_result_output = set_color('valid result', 'blue') + ': \n' + dict2str(valid_result)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                self.tensorboard.add_scalar('Vaild_score', valid_score, epoch_idx)
                self.wandblogger.log_metrics({**valid_result, 'valid_step': valid_step}, head='valid')

                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx, verbose=verbose)
                    self.best_valid_result = valid_result

                if callback_fn:
                    callback_fn(epoch_idx, valid_score)

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

                valid_step+=1

        self._add_hparam_to_tensorboard(self.best_valid_score)
        return self.best_valid_score, self.best_valid_result

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        clean_seq_percent = 100
        try:
            # Note: interaction without item ids
            if 'HSD' in str(self.model):
                scores, clean_seq_percent = self.model.full_sort_predict(interaction.to(self.device))
            else:
                scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i, clean_seq_percent

    def _full_sort_batch_eval_ssd(self, batched_data,best_flag):
        interaction, history_index, positive_u, positive_i = batched_data
        clean_seq_percent = 100
        try:
            # Note: interaction without item ids
            if 'HSD' in str(self.model):
                # scores, clean_seq_percent = self.model.full_sort_predict(interaction.to(self.device),best_flag)
                scores, clean_seq_percent = self.model.full_sort_predict_ssd(interaction.to(self.device),best_flag)
            else:
                scores = self.model.full_sort_predict(interaction.to(self.device))
        except NotImplementedError:
            inter_len = len(interaction)
            new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
            batch_size = len(new_inter)
            new_inter.update(self.item_tensor.repeat(inter_len))
            if batch_size <= self.test_batch_size:
                scores = self.model.predict(new_inter)
            else:
                scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i, clean_seq_percent

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            collections.OrderedDict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            best_flag = True
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.load_other_parameter(checkpoint.get('other_parameter'))
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)
        else:
            best_flag = False
        self.model.eval()

        if isinstance(eval_data, FullSortEvalDataLoader):
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
            # fa = 'hsd' in str(self.model)
            # print('???',fa)
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
            if 'HSD' in str(self.model):
                eval_func = self._full_sort_batch_eval_ssd
            else:
                eval_func = self._full_sort_batch_eval
            if self.item_tensor is None:
                self.item_tensor = eval_data.dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        clean_seq_total = 0
        batch_counter = 0
        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", 'pink'),
            ) if show_progress else eval_data
        )
        for batch_idx, batched_data in enumerate(iter_data):
            if 'HSD' in str(self.model):
                interaction, scores, positive_u, positive_i, clean_seq_percent = eval_func(batched_data,best_flag)
            else:
                interaction, scores, positive_u, positive_i, clean_seq_percent = eval_func(batched_data)
            clean_seq_total += clean_seq_percent
            batch_counter += 1
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(set_color('GPU RAM: ' + get_gpu_usage(self.device), 'yellow'))
            self.eval_collector.eval_batch_collect(scores, interaction, positive_u, positive_i)
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head='eval')

        return result