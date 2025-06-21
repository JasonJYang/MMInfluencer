import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class KGDataset(Dataset):
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

        return user, target_entity, user_id, target_entity_id, label, -1, -1, -1, -1
    

class KGDataLoader():
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
        if not self.save_dir.exists():
            self.save_dir.mkdir()
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

        self.kg, self.entity_num, self.relation_num, \
            self.label_df, self.user_id_list, self.target_entity_id_list, self.id2node = self._process_data()
        self.neg_label_df = self._negative_sampling(self.label_df, self.user_id_list, self.target_entity_id_list, self.id2node)

    def get_kg(self):
        return self.kg
    
    def get_entity_num(self):
        return self.entity_num
    
    def get_relation_num(self):
        return self.relation_num
    
    def get_node2id(self):
        node2id = {v: k for k, v in self.id2node.items()}
        return node2id
    
    def get_id2node(self):
        return self.id2node

    def _load_edges(self):
        self.logger.info('Loading edges...')
        edges_df = pd.read_csv(self.data_dir.joinpath('edges.csv'))
        self.logger.info('Only using edges from biography and txt_jpg...')
        edges_df = edges_df[(edges_df['data_source'] == 'biography') | (edges_df['data_source'] == 'txt_jpg')]
        self.logger.info('{} edges loaded.'.format(edges_df.shape))
        relationship_list = list(set(edges_df['relationship']))
        self.logger.info('Relationships in edges: {}'.format(relationship_list))
        return edges_df
    
    def _clear_edges(self, edges_df):
        self.logger.info('For edges coming from txt_jpg, removing isolated nodes...')
        # count the frequency of edges that contain nodes in txt_jpb_nodes
        edges_df['source_count'] = edges_df['source'].map(edges_df['source'].value_counts())
        edges_df['target_count'] = edges_df['target'].map(edges_df['target'].value_counts())
        txt_jpg_edges_df = edges_df[edges_df['data_source'] == 'txt_jpg']
        self.logger.info('{} edges before removing isolated nodes.'.format(txt_jpg_edges_df.shape))
        txt_jpg_edges_df = txt_jpg_edges_df[(txt_jpg_edges_df['source_count'] > 1) & (txt_jpg_edges_df['target_count'] > 1)]
        self.logger.info('{} edges left after removing isolated nodes.'.format(txt_jpg_edges_df.shape))
        no_txt_jpg_edges_df = edges_df[edges_df['data_source'] != 'txt_jpg']
        edges_df = pd.concat([no_txt_jpg_edges_df, txt_jpg_edges_df], ignore_index=True)
        return edges_df
    
    def _load_nodes(self):
        self.logger.info('Loading nodes...')
        nodes_df = pd.read_csv(self.data_dir.joinpath('nodes.csv'))
        self.logger.info('Only using nodes from biography and txt_jpg...')
        nodes_df = nodes_df[(nodes_df['data_source'] == 'biography') | (nodes_df['data_source'] == 'txt_jpg')]
        self.logger.info('{} nodes loaded.'.format(nodes_df.shape))
        attribute_list = list(set(nodes_df['attribute']))
        self.logger.info('Attributes in nodes: {}'.format(attribute_list))
        return nodes_df
    
    def _load_users_as_edges(self):
        self.logger.info('Loading users...')
        users_df = pd.read_csv(self.data_dir.joinpath('user_nodes.csv'))
        self.logger.info('{} users loaded.'.format(users_df.shape))
        users_df = users_df.drop_duplicates(subset='username')
        self.logger.info('{} unique users loaded.'.format(users_df.shape))
        
        user_nodes_df = users_df[['username']]
        user_nodes_df['attribute'] = 'username'
        user_nodes_df['source_user'] = user_nodes_df['username']
        user_nodes_df['source_post'] = None
        user_nodes_df['data_source'] = 'user'
        user_nodes_df = user_nodes_df.rename(columns={'username': 'name', 'attribute': 'attribute'})
        
        user_edges_df_dict = {'source': [], 'target': [], 'relationship': [],
                             'source_user': [], 'source_post': [], 'data_source': []}
        col_list = ['username', 'hashtag', 'follower_count', 'follower_category', 'interests', 
                    'self_description', 'occupation_or_industry', 'location', 'contact']
        for i, row in users_df.iterrows():
            for col in col_list:
                if pd.notnull(row[col]):
                    user_edges_df_dict['source'].append(row['username'])
                    user_edges_df_dict['target'].append(row[col])
                    user_edges_df_dict['relationship'].append(col)
                    user_edges_df_dict['source_user'].append(row['username'])
                    user_edges_df_dict['source_post'].append(None)
                    user_edges_df_dict['data_source'].append('user')
        user_edges_df = pd.DataFrame(user_edges_df_dict)
        self.logger.info('{} user edges generated.'.format(user_edges_df.shape))

        return user_nodes_df, user_edges_df
    
    def _create_mapping_dict(self, edges_df):
        self.logger.info('Mapping nodes and relationships...')
        node_list = list(set(edges_df['source']) | set(edges_df['target']))
        relationship_list = list(set(edges_df['relationship']))
        node2id = {node: i+1 for i, node in enumerate(node_list)}
        id2node = {v: k for k, v in node2id.items()}
        relationship2id = {relationship: i+1 for i, relationship in enumerate(relationship_list)}
        id2relationship = {v: k for k, v in relationship2id.items()}
        return node2id, id2node, relationship2id, id2relationship
    
    def _create_kg(self, edges_df):
        kg = {}
        for i, row in edges_df.iterrows():
            head = row['source_id']
            tail = row['target_id']
            relation = row['relationship_id']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        self.logger.info('Knowledge graph created.')
        return kg
    
    def _create_label(self, edges_df, user_node_list, target_entity_list):
        # Use multimodal data to get the same labels
        if self.use_multimodal:
            multimodal_edges_df = edges_df[(edges_df['data_source'] == 'txt_jpg') & (edges_df['relationship'] != self.label_relationship)]
            edges_df = edges_df[(edges_df['data_source'] != 'txt_jpg') | (edges_df['relationship'] == self.label_relationship)]

        edges_no_label_df = edges_df[edges_df['relationship'] != self.label_relationship]
        node_list = list(set(edges_no_label_df['source']) | set(edges_no_label_df['target']))
        # node_list = list(set(edges_df['source']) | set(edges_df['target']))
        
        edges_label_df = edges_df[edges_df['relationship'] == self.label_relationship]
        label_df = edges_label_df[['source', 'target', 'relationship']]
        self.logger.info('{} labels generated with unique {} source nodes and {} target nodes.'.format(
            label_df.shape, len(set(label_df['source'])), len(set(label_df['target']))))
        self.logger.info('{} labels left when only keep labels in the source; {} labels left when only keep labels in the target.'.format(
            label_df[label_df['source'].isin(node_list)].shape, label_df[label_df['target'].isin(node_list)].shape))
        label_df = label_df[(label_df['source'].isin(node_list)) & (label_df['target'].isin(node_list))]
        self.logger.info('{} labels generated when only keeping labels with valid edges in the knowledge graph...'.format(
            label_df.shape))
        
        self.logger.info('Only keeping labels with valid users and target entitys...')
        label_organized_df_dict = {'user': [], 'target_entity': [], 'label': []}
        for i, row in label_df.iterrows():
            if row['source'] in user_node_list and row['target'] in target_entity_list:
                label_organized_df_dict['user'].append(row['source'])
                label_organized_df_dict['target_entity'].append(row['target'])
                label_organized_df_dict['label'].append(1)
            elif row['source'] in target_entity_list and row['target'] in user_node_list:
                label_organized_df_dict['user'].append(row['target'])
                label_organized_df_dict['target_entity'].append(row['source'])
                label_organized_df_dict['label'].append(1)
        label_df = pd.DataFrame(label_organized_df_dict)
        self.logger.info('{} labels generated.'.format(label_df.shape))

        if self.use_multimodal:
            edges_no_label_df = pd.concat([edges_no_label_df, multimodal_edges_df], ignore_index=True)
        
        return edges_no_label_df, label_df
    
    def _process_data(self):
        if not self.use_multimodal:
            edge_save_dir = self.save_dir.joinpath('kg_processed_no_multimodal-{}.csv'.format(self.label_relationship))
            label_save_dir = self.save_dir.joinpath('labels_no_multimodal-{}.csv'.format(self.label_relationship))
        else:
            edge_save_dir = self.save_dir.joinpath('kg_processed-{}.csv'.format(self.label_relationship))
            label_save_dir = self.save_dir.joinpath('labels-{}.csv'.format(self.label_relationship))
        
        if edge_save_dir.exists() and label_save_dir.exists() and self.save_dir.joinpath('nodes.csv').exists():
            self.logger.info('Loading processed data...')
            edges_df = pd.read_csv(edge_save_dir)
            label_df = pd.read_csv(label_save_dir)
            nodes_df = pd.read_csv(self.save_dir.joinpath('nodes.csv'))

            user_node_list = list(set(nodes_df[nodes_df['attribute'] == 'username']['name']))
            target_entity_list = list(set(nodes_df[nodes_df['attribute'] == self.target_entity_name]['name']))

            node2id = dict(zip(edges_df['source'], edges_df['source_id']))
            node2id.update(dict(zip(edges_df['target'], edges_df['target_id'])))
            id2node = {v: k for k, v in node2id.items()}
            relationship2id = dict(zip(edges_df['relationship'], edges_df['relationship_id']))
            id2relationship = {v: k for k, v in relationship2id.items()}
            entity_num = len(node2id)
            relation_num = len(relationship2id)
            self.logger.info('Entity num: {}, Relation num: {}'.format(entity_num, relation_num))
        
        else:
            self.logger.info('\nProcessing data...')
            edges_df = self._load_edges()
            nodes_df = self._load_nodes()
            user_nodes_df, user_edges_df = self._load_users_as_edges()
            
            # processing nodes
            self.logger.info('\nProcessing nodes...')
            nodes_df = pd.concat([nodes_df, user_nodes_df], ignore_index=True)
            nodes_df = nodes_df.drop_duplicates(subset=['name'])
            nodes_df.to_csv(self.save_dir.joinpath('nodes.csv'), index=False)
            user_node_list = list(set(nodes_df[nodes_df['attribute'] == 'username']['name']))
            target_entity_list = list(set(nodes_df[nodes_df['attribute'] == self.target_entity_name]['name']))
            self.logger.info('{} nodes processed: {} users and {} target entitys'.format(
                nodes_df.shape, len(user_node_list), len(target_entity_list)))
            
            # processing edges
            self.logger.info('\nProcessing edges...')
            edges_df = pd.concat([edges_df, user_edges_df], ignore_index=True)
            edges_df = edges_df.drop_duplicates(subset=['source', 'target', 'relationship'])
            edges_df = self._clear_edges(edges_df)
            if not self.use_multimodal:
                self.logger.info('Dropping multimodal data...')
                edges_df = edges_df[(edges_df['data_source'] != 'txt_jpg') | (edges_df['relationship'] == self.label_relationship)]
            else:
                self.logger.info('Using multimodal data...')
            edges_df, label_df = self._create_label(edges_df, user_node_list, target_entity_list)
            
            node_list = list(set(edges_df['source']) | set(edges_df['target']))
            self.logger.info('{} edges processed: {} nodes and {} relationships'.format(
                edges_df.shape, len(node_list), len(list(set(edges_df['relationship'])))))
                
            self.logger.info('\nMapping nodes and relationships...')
            node2id, id2node, relationship2id, id2relationship = self._create_mapping_dict(edges_df)
            entity_num = len(node2id)
            relation_num = len(relationship2id)
            self.logger.info('Entity num: {}, Relation num: {}'.format(entity_num, relation_num))
            edges_df['source_id'] = edges_df['source'].map(node2id)
            edges_df['target_id'] = edges_df['target'].map(node2id)
            edges_df['relationship_id'] = edges_df['relationship'].map(relationship2id)
            label_df['user_id'] = label_df['user'].map(node2id)
            label_df['target_entity_id'] = label_df['target_entity'].map(node2id)
            label_df = label_df[['user', 'target_entity', 'user_id', 'target_entity_id', 'label']]

            self.logger.info('\nSaving processed data...')
            edges_df.to_csv(edge_save_dir, index=False)
            label_df.to_csv(label_save_dir, index=False)

        self.logger.info('\nOnly keeping valid users and target entity...')
        user_id_list = [node2id[node] for node in user_node_list if node in node2id]
        target_entity_id_list = [node2id[node] for node in target_entity_list if node in node2id]
        self.logger.info('{} valid users and {} valid target entity.'.format(
            len(user_id_list), len(target_entity_id_list)))
        sparsity = 1 - len(edges_df) / (entity_num * entity_num)
        self.logger.info('Sparsity: {:.4f}'.format(sparsity))

        self.logger.info('\nCreating knowledge graph...')
        kg = self._create_kg(edges_df)

        self.logger.info('Data processed.\n')

        return kg, entity_num, relation_num, label_df, user_id_list, target_entity_id_list, id2node
    
    def _negative_sampling(self, label_df, user_id_list, target_entity_id_list, id2node):
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
        neg_label_df['user'] = neg_label_df['user_id'].map(id2node)
        neg_label_df['target_entity'] = neg_label_df['target_entity_id'].map(id2node)
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
        train_dataset = KGDataset(train_df)
        valid_dataset = KGDataset(valid_df)
        test_dataset = KGDataset(test_df)

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
            train_dataset = KGDataset(train_df)
            valid_dataset = KGDataset(valid_df)
            test_dataset = KGDataset(test_df)

            self.logger.info('Creating dataloaders...')
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
            dataloader_list.append((train_loader, valid_loader, test_loader))

        return dataloader_list
    
    def get_dataloader_inference_kth_fold(self, k, resume_dir, node2id, id2node):
        self.logger.info('Creating inference dataloader for fold {}...'.format(k))
        resume_dir = Path(resume_dir)
        binary_result_df = pd.read_csv(resume_dir.joinpath('test_result_K{}.csv'.format(k)))
        positive_result_df = binary_result_df[binary_result_df['target'] == 1]
        positive_result_df['user_id'] = positive_result_df['user'].map(node2id)
        positive_result_df['target_entity_id'] = positive_result_df['product'].map(node2id)
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
            left_target_entity_list = list(set(self.target_entity_id_list) - set(pos_target_entity_list))
            inference_df_dict['user_id'] += [user_id] * len(left_target_entity_list)
            inference_df_dict['target_entity_id'] += left_target_entity_list
            inference_df_dict['label'] += [0] * len(left_target_entity_list)
        inference_df = pd.DataFrame(inference_df_dict)
        inference_df['user'] = inference_df['user_id'].map(id2node)
        inference_df['target_entity'] = inference_df['target_entity_id'].map(id2node)
        self.logger.info('{} inference samples generated.'.format(inference_df.shape))
        inference_dataset = KGDataset(inference_df)
        inference_loader = DataLoader(inference_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return inference_loader

    def get_dataloader_node2vec(self):
        self.logger.info('Creating dataloader for node2vec...')
        label_df = pd.concat([self.label_df, self.neg_label_df], ignore_index=True)
        label_df = label_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.logger.info('label distribution: {}'.format(label_df['label'].value_counts()))

        train_dataset = KGDataset(label_df)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        return train_loader
    
    def get_dataloader_inference_node2vec(self, node2id, id2node):
        positive_result_df = self.label_df
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
            left_target_entity_list = list(set(self.target_entity_id_list) - set(pos_target_entity_list))
            inference_df_dict['user_id'] += [user_id] * len(left_target_entity_list)
            inference_df_dict['target_entity_id'] += left_target_entity_list
            inference_df_dict['label'] += [0] * len(left_target_entity_list)
        inference_df = pd.DataFrame(inference_df_dict)
        inference_df['user'] = inference_df['user_id'].map(id2node)
        inference_df['target_entity'] = inference_df['target_entity_id'].map(id2node)
        self.logger.info('{} inference samples generated.'.format(inference_df.shape))
        inference_dataset = KGDataset(inference_df)
        inference_loader = DataLoader(inference_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return inference_loader
