import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CFDataset(Dataset):
    def __init__(self, label_df):
        self.label_df = label_df
        
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        user = self.label_df.iloc[idx]['user']
        target_entity = self.label_df.iloc[idx]['target_entity']
        user_id = self.label_df.iloc[idx]['user_id']
        target_entity_id = self.label_df.iloc[idx]['target_entity_id']
        label = self.label_df.iloc[idx]['label']

        return user, target_entity, user_id, target_entity_id, label
    

class CFDataLoader():
    def __init__(self, logger,
                       data_dir, 
                       use_multimodal=True,
                       label_relationship='suitable_category',
                       target_entity_name='product_category',
                       batch_size=32, 
                       seed=42, 
                       shuffle=True, 
                       validation_split=0.1, 
                       test_split=0.2, 
                       num_workers=1):
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.save_dir = self.data_dir.parent.joinpath('processed')
        self.label_relationship = label_relationship
        self.target_entity_name = target_entity_name
        self.use_multimodal = use_multimodal
        self.logger.info('Using multimodal data: {} with {} as label relationship and {} as target entity name'.format(
            self.use_multimodal, self.label_relationship, self.target_entity_name))
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.test_split = test_split
        self.num_workers = num_workers

        random.seed(self.seed)

        self.label_df, self.user_id_list, self.target_entity_id_list, \
            self.user_num, self.target_entity_num, self.user2id, self.target_entity2id = self._process_data()
        self.id2user = {v: k for k, v in self.user2id.items()}
        self.id2target_entity = {v: k for k, v in self.target_entity2id.items()}
        self.neg_label_df = self._negative_sampling(self.label_df, self.user_id_list, self.target_entity_id_list, 
                                                    self.id2user, self.id2target_entity)

    def get_user_num(self):
        return self.user_num

    def get_entity_num(self):
        return self.target_entity_num
    
    def _process_data(self):
        if not self.use_multimodal:
            label_save_dir = self.save_dir.joinpath('labels_no_multimodal-{}.csv'.format(self.label_relationship))
        else:
            label_save_dir = self.save_dir.joinpath('labels-{}.csv'.format(self.label_relationship))
        
        if label_save_dir.exists():
            self.logger.info('Loading processed data...')
            label_df = pd.read_csv(label_save_dir)

            user_node_list = list(set(label_df['user']))
            target_entity_list = list(set(label_df['target_entity']))

            user2id = {k: v for v, k in enumerate(user_node_list)}
            target_entity2id = {k: v for v, k in enumerate(target_entity_list)}
            label_df['user_id'] = label_df['user'].map(user2id)
            label_df['target_entity_id'] = label_df['target_entity'].map(target_entity2id)
            self.logger.info('Loaded {} labels with {} users and {} target entities.'.format(
                label_df.shape[0], len(user_node_list), len(target_entity_list)))        
            
            user_id_list = list(set(label_df['user_id']))
            target_entity_id_list = list(set(label_df['target_entity_id']))
            user_num = len(user_id_list)
            target_entity_num = len(target_entity_id_list)

        return label_df, user_id_list, target_entity_id_list, user_num, target_entity_num, user2id, target_entity2id
    
    def _negative_sampling(self, label_df, user_id_list, target_entity_id_list, id2user, id2target_entity):
        self.logger.info('Negative sampling for each user...')
        neg_label_df_dict = {'user_id': [], 'target_entity_id': [], 'label': []}
        for user in list(set(label_df['user_id'])):
            user_label_df = label_df[label_df['user_id'] == user]
            neg_product_category_list = list(set(target_entity_id_list) - set(user_label_df['target_entity_id']))
            neg_product_category_list = random.sample(neg_product_category_list, len(user_label_df))
            neg_label_df_dict['user_id'] += [user] * len(neg_product_category_list)
            neg_label_df_dict['target_entity_id'] += neg_product_category_list
            neg_label_df_dict['label'] += [0] * len(neg_product_category_list)
        neg_label_df = pd.DataFrame(neg_label_df_dict)
        neg_label_df['user'] = neg_label_df['user_id'].map(id2user)
        neg_label_df['target_entity'] = neg_label_df['target_entity_id'].map(id2target_entity)
        self.logger.info('{} negative samples generated.'.format(neg_label_df.shape))
        return neg_label_df
    
    def get_dataloader(self):
        self.logger.info('Creating dataloader...')
        label_df = pd.concat([self.label_df, self.neg_label_df], ignore_index=True)
        label_df = label_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.logger.info('label distribution: {}'.format(label_df['label'].value_counts()))
        train_num = int(len(label_df) * (1 - self.test_split - self.validation_split))
        test_num = int(len(label_df) * self.test_split)
        train_df = label_df[:train_num]
        test_df = label_df[train_num:train_num+test_num]
        valid_df = label_df[train_num+test_num:]
        self.logger.info('Train: {}, Valid: {}, Test: {}'.format(train_df.shape, valid_df.shape, test_df.shape))

        self.logger.info('Creating datasets...')
        train_dataset = CFDataset(train_df)
        valid_dataset = CFDataset(valid_df)
        test_dataset = CFDataset(test_df)

        self.logger.info('Creating dataloaders...')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

        return train_loader, valid_loader, test_loader
    
    def get_dataloader_kfold(self, K):
        self.logger.info('Creating dataloader for {}-fold...'.format(K))
        label_df = pd.concat([self.label_df, self.neg_label_df], ignore_index=True)
        label_df = label_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.logger.info('label distribution: {}'.format(label_df['label'].value_counts()))

        label_index_list = list(range(len(label_df)))
        np.random.seed(self.seed)
        np.random.shuffle(label_index_list)
        fold_size = len(label_index_list) // K
        fold_index_list = [label_index_list[i*fold_size: (i+1)*fold_size] for i in range(K)]
        if len(fold_index_list) % K != 0:
            fold_index_list[-1] += label_index_list[K*fold_size:]
    
        kfold_index_dict = {}
        for fold_idx in range(K):
            test_index_list = fold_index_list[fold_idx]
            train_index_list = []
            for i in range(K):
                if i != fold_idx:
                    train_index_list += fold_index_list[i]
            # 10% of training data is used for validation
            np.random.shuffle(train_index_list)
            valid_index_list = train_index_list[: int(len(train_index_list)*0.1)]
            train_index_list = train_index_list[int(len(train_index_list)*0.1): ]
            kfold_index_dict[fold_idx] = (train_index_list, valid_index_list, test_index_list)

        dataloader_list = []
        for fold_idx in range(K):
            train_index_list, valid_index_list, test_index_list = kfold_index_dict[fold_idx]
            train_df = label_df.iloc[train_index_list]
            valid_df = label_df.iloc[valid_index_list]
            test_df = label_df.iloc[test_index_list]
            self.logger.info('Fold {}: Train: {}, Valid: {}, Test: {}'.format(fold_idx, train_df.shape, valid_df.shape, test_df.shape))

            self.logger.info('Creating datasets...')
            train_dataset = CFDataset(train_df)
            valid_dataset = CFDataset(valid_df)
            test_dataset = CFDataset(test_df)

            self.logger.info('Creating dataloaders...')
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            dataloader_list.append((train_loader, valid_loader, test_loader))

        return dataloader_list
    
    def get_dataloader_inference_kth_fold(self, k, resume_dir, user2id, target_entity2id,
                                          id2user, id2target_entity):
        self.logger.info('Creating inference dataloader for fold {}...'.format(k))
        resume_dir = Path(resume_dir)
        binary_result_df = pd.read_csv(resume_dir.joinpath('test_result_K{}.csv'.format(k)))
        positive_result_df = binary_result_df[binary_result_df['target'] == 1]
        positive_result_df['user_id'] = positive_result_df['user'].map(user2id)
        positive_result_df['target_entity_id'] = positive_result_df['target_entity'].map(target_entity2id)
        # for each user, go through all the products in the product list
        inference_df_dict = {'user_id': [], 'target_entity_id': [], 'label': []}
        for user_id in list(set(positive_result_df['user_id'])):
            # positive
            user_result_df = positive_result_df[positive_result_df['user_id'] == user_id]
            pos_target_entity_list = list(set(user_result_df['target_entity_id']))
            inference_df_dict['user_id'] += [user_id] * len(pos_target_entity_list)
            inference_df_dict['target_entity_id'] += pos_target_entity_list
            inference_df_dict['label'] += [1] * len(pos_target_entity_list)
            # negative
            left_target_entity_with_name_list = list(set(self.target_entity_id_list) - set(pos_target_entity_list))
            inference_df_dict['user_id'] += [user_id] * len(left_target_entity_with_name_list)
            inference_df_dict['target_entity_id'] += left_target_entity_with_name_list
            inference_df_dict['label'] += [0] * len(left_target_entity_with_name_list)
        inference_df = pd.DataFrame(inference_df_dict)
        inference_df['user'] = inference_df['user_id'].map(id2user)
        # for target_entity not in the target_entity2id, use 'unknown' as the name
        id2target_entity_update = {k: v if k in self.target_entity_id_list else 'unknown' for k, v in id2target_entity.items()}
        inference_df['target_entity'] = inference_df['target_entity_id'].map(lambda x: id2target_entity_update.get(x, 'unknown'))
        self.logger.info('{} inference samples generated.'.format(inference_df.shape))
        inference_dataset = CFDataset(inference_df)
        inference_loader = DataLoader(inference_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return inference_loader
