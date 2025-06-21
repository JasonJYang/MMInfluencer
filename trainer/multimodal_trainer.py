import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class for Multimodal Classifier model
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
        for batch_idx, (bio_embeddings, post_text_embeddings, post_images, user_id, target_entity_id) in enumerate(self.train_data_loader):
            bio_embeddings = bio_embeddings.to(self.device)
            post_text_embeddings = post_text_embeddings.to(self.device)
            post_images = post_images.to(self.device)
            user_id, target_entity_id = user_id.to(self.device), target_entity_id.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(bio_embeddings, post_text_embeddings, post_images)
            loss = self.criterion(output, target_entity_id.long())
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            
            for met in self.metric_fns:
                try:
                    self.train_metrics.update(met.__name__, 
                        met(output.cpu().detach().numpy(), target_entity_id.cpu().detach().numpy()))
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
            for batch_idx, (bio_embeddings, post_text_embeddings, post_images, user_id, target_entity_id) in enumerate(self.valid_data_loader):
                bio_embeddings = bio_embeddings.to(self.device)
                post_text_embeddings = post_text_embeddings.to(self.device)
                post_images = post_images.to(self.device)
                user_id, target_entity_id = user_id.to(self.device), target_entity_id.to(self.device)

                output = self.model(bio_embeddings, post_text_embeddings, post_images)
                loss = self.criterion(output, target_entity_id.long())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                
                y_pred.append(output.cpu().detach().numpy())
                y_true.append(target_entity_id.cpu().detach().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        for met in self.metric_fns:
            self.valid_metrics.update(met.__name__, met(y_pred, y_true))
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def test(self, id2user, id2entity):
        """
        Test the trained model
        """
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []
        user_list, target_entity_list = [], []
        with torch.no_grad():
            for batch_idx, (bio_embeddings, post_text_embeddings, post_images, user_id, target_entity_id) in enumerate(self.test_data_loader):
                bio_embeddings = bio_embeddings.to(self.device)
                post_text_embeddings = post_text_embeddings.to(self.device)
                post_images = post_images.to(self.device)
                user_id, target_entity_id = user_id.to(self.device), target_entity_id.to(self.device)

                output = self.model(bio_embeddings, post_text_embeddings, post_images)
                loss = self.criterion(output, target_entity_id.long())

                # apply softmax to output
                output = torch.softmax(output, dim=1)
                y_pred.append(output.cpu().detach().numpy())
                y_true.append(target_entity_id.cpu().detach().numpy())

                # Convert tensor to list for dataframe
                user_list.extend([id2user[u] for u in user_id.cpu().numpy().tolist()])
                target_entity_list.extend([id2entity[e] for e in target_entity_id.cpu().numpy().tolist()])

                batch_size = user_id.shape[0]
                total_loss += loss.item() * batch_size

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        test_metrics = {'loss': total_loss / len(self.test_data_loader.dataset)}     
        for metric in self.metric_fns:
            test_metrics[metric.__name__] = metric(y_pred, y_true)

        result_df_dict = {'user': [], 'target_entity': [], 'target': [], 'prediction': []}
        target_entity_list = [id2entity[i] for i in range(len(id2entity))]
        for i, user in enumerate(user_list):
            user_pred = y_pred[i].tolist()
            result_df_dict['target_entity'] += target_entity_list
            result_df_dict['prediction'] += user_pred
            result_df_dict['user'] += [user] * len(target_entity_list)
            result_df_dict['target'] += [id2entity[y_true[i]]] * len(target_entity_list)

        result_df = pd.DataFrame(result_df_dict)

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