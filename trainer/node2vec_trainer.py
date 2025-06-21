import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
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
        total_loss = 0
        for batch_idx, (pos_rw, neg_rw) in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()
            loss = self.model.node2vec.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
        
        log = {'train': {'loss': total_loss / len(self.train_data_loader)}}        
        return log
    
    def test(self):
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []
        user_list, target_entity_list = [], []
        with torch.no_grad():
            for batch_idx, (user_name, target_entity_name, user_id, target_entity_id, target, \
                user_entities, user_relationships, target_entity_entities, target_entity_relationships) \
                    in tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader)):
                user, target_entity = user_id.to(self.device), target_entity_id.to(self.device)
                user_entities, user_relationships = user_entities.to(self.device), user_relationships.to(self.device)
                target_entity_entities, target_entity_relationships = target_entity_entities.to(self.device), target_entity_relationships.to(self.device)
                target = target.to(self.device).float()

                output = self.model(user, user_entities, user_relationships, 
                                    target_entity, target_entity_entities, target_entity_relationships)
                loss = self.criterion(output, target.float())

                output = torch.sigmoid(output)
                y_pred.append(output.cpu().detach().numpy())
                y_true.append(target.cpu().detach().numpy())
                user_list += user_name
                target_entity_list += target_entity_name

                batch_size = user.shape[0]
                total_loss += loss.item() * batch_size

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        test_metrics = {'loss': total_loss / len(self.test_data_loader.dataset)}     
        for metric in self.metric_fns:
            test_metrics[metric.__name__] = metric(y_pred, y_true)
        
        result_df = pd.DataFrame({'user': user_list, 'target_entity': target_entity_list, 'target': y_true, 'prediction': y_pred})

        return test_metrics, result_df
    
    def test_zero_shot(self):
        self.model.eval()
        y_true, y_pred = [], []
        user_list, target_entity_list = [], []
        with torch.no_grad():
            for batch_idx, (user_name, target_entity_name, user_id, target_entity_id, target, \
                user_entities, user_relationships, target_entity_entities, target_entity_relationships) \
                    in tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader)):
                user, target_entity = user_id.to(self.device), target_entity_id.to(self.device)
                user_entities, user_relationships = user_entities.to(self.device), user_relationships.to(self.device)
                target_entity_entities, target_entity_relationships = target_entity_entities.to(self.device), target_entity_relationships.to(self.device)
                target = target.to(self.device).float()

                output = self.model(user, user_entities, user_relationships, 
                                    target_entity, target_entity_entities, target_entity_relationships)

                output = torch.sigmoid(output)
                y_pred.append(output.cpu().detach().numpy())
                y_true.append(target.cpu().detach().numpy())
                user_list += user_name
                target_entity_list += target_entity_name

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        test_metrics = {}     
        for metric in self.metric_fns:
            if metric.__name__ == 'accuracy':
                test_metrics[metric.__name__] = metric(y_pred, y_true)
        
        result_df = pd.DataFrame({'user': user_list, 'target_entity': target_entity_list, 'target': y_true, 'prediction': y_pred})

        return test_metrics, result_df

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_data_loader, 'n_samples'):
            current = batch_idx * self.train_data_loader.batch_size
            total = self.train_data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
