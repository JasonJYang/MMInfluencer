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
        self.train_metrics.reset()
        for batch_idx, (_, _, user_id, target_entity_id, target, \
                user_entities, user_relationships, target_entity_entities, target_entity_relationships) in enumerate(self.train_data_loader):
            user, target_entity = user_id.to(self.device), target_entity_id.to(self.device)
            user_entities, user_relationships = user_entities.to(self.device), user_relationships.to(self.device)
            target_entity_entities, target_entity_relationships = target_entity_entities.to(self.device), target_entity_relationships.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(user, user_entities, user_relationships, 
                                target_entity, target_entity_entities, target_entity_relationships)
            loss = self.criterion(output, target.float())
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            output = torch.sigmoid(output)
            for met in self.metric_fns:
                try:
                    self.train_metrics.update(met.__name__, 
                        met(output.cpu().detach().numpy(), target.cpu().detach().numpy()))
                except:
                    self.train_metrics.update(met.__name__, 0)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        log['train'] = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log['validation'] = {'val_'+k : v for k, v in val_log.items()}

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
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_idx, (_, _, user_id, target_entity_id, target, \
                    user_entities, user_relationships, target_entity_entities, target_entity_relationships) in enumerate(self.valid_data_loader):
                user, target_entity = user_id.to(self.device), target_entity_id.to(self.device)
                user_entities, user_relationships = user_entities.to(self.device), user_relationships.to(self.device)
                target_entity_entities, target_entity_relationships = target_entity_entities.to(self.device), target_entity_relationships.to(self.device)
                target = target.to(self.device).float()

                output = self.model.inference_zero_shot(user, user_entities, user_relationships, 
                                                         target_entity, target_entity_entities, target_entity_relationships)
                # loss = self.criterion(output, target.float())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # self.valid_metrics.update('loss', loss.item())
                
                output = torch.sigmoid(output)
                y_pred.append(output.cpu().detach().numpy())
                y_true.append(target.cpu().detach().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        for met in self.metric_fns:
            if met.__name__ == 'accuracy':
                self.valid_metrics.update(met.__name__, met(y_pred, y_true))
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def test(self):
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []
        user_list, product_list = [], []
        with torch.no_grad():
            for batch_idx, (user_name, product_name, user_id, target_entity_id, target, \
                    user_entities, user_relationships, target_entity_entities, target_entity_relationships) \
                        in tqdm(enumerate(self.test_data_loader), total=len(self.test_data_loader)):
                user, target_entity = user_id.to(self.device), target_entity_id.to(self.device)
                user_entities, user_relationships = user_entities.to(self.device), user_relationships.to(self.device)
                target_entity_entities, target_entity_relationships = target_entity_entities.to(self.device), target_entity_relationships.to(self.device)
                target = target.to(self.device).float()

                output = self.model.inference_zero_shot(user, user_entities, user_relationships, 
                                                         target_entity, target_entity_entities, target_entity_relationships)
                # loss = self.criterion(output, target.float())

                output = torch.sigmoid(output)
                y_pred.append(output.cpu().detach().numpy())
                y_true.append(target.cpu().detach().numpy())
                user_list += user_name
                product_list += product_name

                batch_size = user.shape[0]
                # total_loss += loss.item() * batch_size

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        # test_metrics = {'loss': total_loss / len(self.test_data_loader.dataset)}     
        test_metrics = {}
        for metric in self.metric_fns:
            if metric.__name__ == 'accuracy':
                test_metrics[metric.__name__] = metric(y_pred, y_true)
        
        result_df = pd.DataFrame({'user': user_list, 'target_entity': product_list, 'target': y_true, 'prediction': y_pred})

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

    def update_model_for_new_user(self, new_user_data):
        new_nodes, new_edges = self._process_new_user_data(new_user_data)
        self.model.update_graph(new_nodes, new_edges)
        
        # Optionally, perform a few training steps to fine-tune the model
        self.model.train()
        for _ in range(5):  # Adjust the number of steps as needed
            self._train_step(new_user_data)

    def _process_new_user_data(self, new_user_data):
        # Process new user data to extract new nodes and edges
        # This is a placeholder - implement according to your data structure
        new_nodes = []
        new_edges = {}
        # ... process new_user_data ...
        return new_nodes, new_edges

    def _train_step(self, data):
        user, product, target = data
        user, product = user.to(self.device), product.to(self.device)
        target = target.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(user, product)
        loss = self.criterion(output, target.float())
        loss.backward()
        self.optimizer.step()
