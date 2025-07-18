import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class MFTrainer(BaseTrainer):
    """
    Trainer class for collaborative filtering models
    """
    def __init__(self, model, criterion, metric_fns, optimizer, config,
                 train_data_loader, valid_data_loader=None, test_data_loader=None,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_fns, optimizer, config)
        self.config = config
        self.train_data_loader = train_data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_data_loader)
        else:
            # iteration-based training
            self.train_data_loader = inf_loop(train_data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_fns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
            
        # Matrix factorization training
        total_loss = 0
        for batch_idx, (_, _, user, product, target) in enumerate(self.train_data_loader):
            user, product, target = user.to(self.device), product.to(self.device), target.to(self.device).float()

            self.optimizer.zero_grad()
            output = self.model(user, product)
            loss = self.criterion(output, target)

            # Add regularization if available
            if hasattr(self.model, 'regularization_loss'):
                reg_loss = self.model.regularization_loss()
                loss += reg_loss
                
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.train_metrics.update('loss', loss.item())
            
            for met in self.metric_fns:
                try:
                    self.train_metrics.update(met.__name__, 
                        met(output.cpu().detach().numpy(), target.cpu().detach().numpy()))
                except:
                    continue

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            total_loss += loss.item()

        log = self.train_metrics.result()
        
        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (_, _, user, product, target) in enumerate(self.valid_data_loader):
                user, product, target = user.to(self.device), product.to(self.device), target.to(self.device).float()

                output = self.model(user, product)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_fns:
                    try:
                        self.valid_metrics.update(met.__name__, 
                            met(output.cpu().detach().numpy(), target.cpu().detach().numpy()))
                    except:
                        continue

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def test(self):
        """
        Test the model
        :return: A log that contains information about test
        """
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []
        user_list, target_entity_list = [], []
        with torch.no_grad():
            for batch_idx, (user_name, product_name, user, product, target) in enumerate(self.test_data_loader):
                user, product, target = user.to(self.device), product.to(self.device), target.to(self.device).float()

                output = self.model(user, product)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                
                # Store outputs and labels for metrics
                y_true.append(target.cpu().detach().numpy())
                y_pred.append(output.cpu().detach().numpy())
                user_list += user_name
                target_entity_list += product_name

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        test_metrics = {'loss': total_loss / len(self.test_data_loader.dataset)}     
        for metric in self.metric_fns:
            test_metrics[metric.__name__] = metric(y_pred, y_true)

        result_df = pd.DataFrame({'user': user_list, 'target_entity': target_entity_list, 
                                  'target': y_true, 'prediction': y_pred})

        return test_metrics, result_df
    
    def recommendation_inference(self):
        self.model.eval()
        y_true, y_pred = [], []
        user_list, target_entity_list = [], []
        with torch.no_grad():
            for batch_idx, (user_name, target_entity_name, user_id, target_entity_id, target) in \
                        tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader)):
                user, target_entity = user_id.to(self.device), target_entity_id.to(self.device)
                target = target.to(self.device).float()

                output = self.model(user, target_entity)
                
                y_pred.append(output.cpu().detach().numpy())
                y_true.append(target.cpu().detach().numpy())
                user_list += user_name
                target_entity_list += target_entity_name

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        
        result_df = pd.DataFrame({'user': user_list, 'target_entity': target_entity_list, 
                                  'target': y_true, 'prediction': y_pred})

        return result_df
    
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = len(self.train_data_loader)
        return base.format(current, total, 100.0 * current / total)