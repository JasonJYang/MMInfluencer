import torch
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
    

class KGDataset(Dataset):
    def __init__(self, data_dict, n_neighbor):
        self.label_df = data_dict['label']
        self.userid2entityid = data_dict['userid2entityid']
        self.userid2relationship = data_dict['userid2relationship']
        self.target_entityid2entityid = data_dict['target_entityid2entityid']
        self.target_entityid2relationship = data_dict['target_entityid2relationship']
        self.n_neighbor = n_neighbor

    def neighbor_sampling(self, entities, relations):
        entity_index_list = list(range(len(entities)))
        if len(entity_index_list) >= self.n_neighbor:
            neighbors = random.sample(entity_index_list, self.n_neighbor)
        else:
            neighbors = random.choices(entity_index_list, k=self.n_neighbor)
        return torch.LongTensor([entities[idx] for idx in neighbors]), torch.LongTensor([relations[idx] for idx in neighbors])

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, idx):
        user = self.label_df.iloc[idx]['user']
        target_entity = self.label_df.iloc[idx]['target_entity']
        user_id = self.label_df.iloc[idx]['user_id']
        target_entity_id = self.label_df.iloc[idx]['target_entity_id']
        label = self.label_df.iloc[idx]['label']

        user_entities, user_relationships = self.neighbor_sampling(self.userid2entityid[user_id], 
                                                                   self.userid2relationship[user_id])
        target_entity_entities, target_entity_relationships = self.neighbor_sampling(self.target_entityid2entityid[target_entity_id], 
                                                                                     self.target_entityid2relationship[target_entity_id])

        return user, target_entity, user_id, target_entity_id, label, user_entities, user_relationships, target_entity_entities, target_entity_relationships


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
                       num_workers=1,
                       K=5,
                       n_neighbor=5):
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.save_dir = self.data_dir.parent.joinpath('zero_shot_processed')
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
        self.n_neighbor = n_neighbor

        random.seed(self.seed)
        np.random.seed(self.seed)

        self._process_data_kfold(K)

    def get_kg(self, data_train_dict):
        return data_train_dict['kg']

    def get_entity_num(self, data_train_dict):
        return len(data_train_dict['node2id'])
    
    def get_relation_num(self, data_train_dict):
        return len(data_train_dict['relationship2id'])

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
        if not self.use_multimodal:
            self.logger.info('Dropping multimodal data...')
            edges_df = edges_df[(edges_df['data_source'] != 'txt_jpg') | (edges_df['relationship'] == self.label_relationship)]
        else:
            self.logger.info('Using multimodal data...')
            # Use multimodal data to get the same labels
            multimodal_edges_df = edges_df[(edges_df['data_source'] == 'txt_jpg') & (edges_df['relationship'] != self.label_relationship)]
            edges_df = edges_df[(edges_df['data_source'] != 'txt_jpg') | (edges_df['relationship'] == self.label_relationship)]

        edges_no_label_df = edges_df[edges_df['relationship'] != self.label_relationship]
        node_list = list(set(edges_no_label_df['source']) | set(edges_no_label_df['target']))
        
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
    
    def _create_kg_label_train(self, nodes_df, edges_df, user_nodes_train_df, user_edges_train_df):
        self.logger.info('Preparing edges for training...')
        # create knowledge graph using train users
        self.logger.info('Processing nodes...')
        nodes_df = pd.concat([nodes_df, user_nodes_train_df], ignore_index=True)
        nodes_df = nodes_df.drop_duplicates(subset=['name'])
        user_node_train_list = list(set(nodes_df[nodes_df['attribute'] == 'username']['name']))
        target_entity_train_list = list(set(nodes_df[nodes_df['attribute'] == self.target_entity_name]['name']))
        nodes_other_df = nodes_df[~nodes_df['source_user'].isin(user_node_train_list)]
        nodes_train_user_df = nodes_df[nodes_df['source_user'].isin(user_node_train_list)]
        nodes_df = pd.concat([nodes_train_user_df, nodes_other_df], ignore_index=True)
        self.logger.info('{} nodes processed: {} users and {} target entitys'.format(
            nodes_df.shape, len(user_node_train_list), len(target_entity_train_list)))
        
        # processing edges
        self.logger.info('Processing edges...')
        edges_df = pd.concat([edges_df, user_edges_train_df], ignore_index=True)
        edges_df = edges_df.drop_duplicates(subset=['source', 'target', 'relationship'])
        edges_other_df = edges_df[~edges_df['source_user'].isin(user_node_train_list)]
        edge_train_user_df = edges_df[edges_df['source_user'].isin(user_node_train_list)]
        edges_df = pd.concat([edge_train_user_df, edges_other_df], ignore_index=True)
        edges_df = self._clear_edges(edges_df)
        edges_df, label_train_df = self._create_label(edges_df, user_node_train_list, target_entity_train_list)

        node_list = list(set(edges_df['source']) | set(edges_df['target']))
        self.logger.info('{} edges processed: {} nodes and {} relationships'.format(
            edges_df.shape, len(node_list), len(list(set(edges_df['relationship'])))))
            
        self.logger.info('Mapping nodes and relationships...')
        node2id, id2node, relationship2id, id2relationship = self._create_mapping_dict(edges_df)
        entity_num = len(node2id)
        relation_num = len(relationship2id)
        self.logger.info('Entity num: {}, Relation num: {}'.format(entity_num, relation_num))
        edges_df['source_id'] = edges_df['source'].map(node2id)
        edges_df['target_id'] = edges_df['target'].map(node2id)
        edges_df['relationship_id'] = edges_df['relationship'].map(relationship2id)
        label_train_df['user_id'] = label_train_df['user'].map(node2id)
        label_train_df['target_entity_id'] = label_train_df['target_entity'].map(node2id)
        label_train_df = label_train_df[['user', 'target_entity', 'user_id', 'target_entity_id', 'label']]

        user_train_id_list = [node2id[node] for node in user_node_train_list if node in node2id]
        target_entity_train_id_list = [node2id[node] for node in target_entity_train_list if node in node2id]

        # create user's entities, relationships for training        
        user2entity_dict = {user: [] for user in user_train_id_list}
        user2relationship_dict = {user: [] for user in user_train_id_list}
        target_entity2entity_dict = {entity: [] for entity in target_entity_train_id_list}
        target_entity2relationship_dict = {entity: [] for entity in target_entity_train_id_list}
        for i, row in edges_df.iterrows():
            if row['source_id'] in user_train_id_list:
                user2entity_dict[row['source_id']].append(row['target_id'])
                user2relationship_dict[row['source_id']].append(row['relationship_id'])
            if row['target_id'] in user_train_id_list:
                user2entity_dict[row['target_id']].append(row['source_id'])
                user2relationship_dict[row['target_id']].append(row['relationship_id'])
            if row['source_id'] in target_entity_train_id_list:
                target_entity2entity_dict[row['source_id']].append(row['target_id'])
                target_entity2relationship_dict[row['source_id']].append(row['relationship_id'])
            if row['target_id'] in target_entity_train_id_list:
                target_entity2entity_dict[row['target_id']].append(row['source_id'])
                target_entity2relationship_dict[row['target_id']].append(row['relationship_id'])
        # remove users and entities that do not have valid relationships
        user2entity_dict = {user: value for user, value in user2entity_dict.items() if len(value) > 0}
        user2relationship_dict = {user: value for user, value in user2relationship_dict.items() if len(value) > 0}
        target_entity2entity_dict = {entity: value for entity, value in target_entity2entity_dict.items() if len(value) > 0}
        target_entity2relationship_dict = {entity: value for entity, value in target_entity2relationship_dict.items() if len(value) > 0}
        user_train_id_list = list(user2entity_dict.keys())
        target_entity_train_id_list = list(target_entity2entity_dict.keys())
        label_train_df = label_train_df[(label_train_df['user_id'].isin(user_train_id_list)) &
                                        (label_train_df['target_entity_id'].isin(target_entity_train_id_list))]
        self.logger.info('Only keeping labels with valid users and target entitys...')
        self.logger.info('{} labels left.'.format(label_train_df.shape))

        # negative sampling
        neg_label_df = self._negative_sampling(label_train_df, target_entity_train_id_list, id2node)
        label_train_df = pd.concat([label_train_df, neg_label_df], ignore_index=True)

        self.logger.info('\nOnly keeping valid users and target entity...')
        user_id_train_list = [node2id[node] for node in user_node_train_list if node in node2id]
        target_entity_id_train_list = [node2id[node] for node in target_entity_train_list if node in node2id]
        self.logger.info('{} valid users and {} valid target entity.'.format(
            len(user_id_train_list), len(target_entity_id_train_list)))
        sparsity = 1 - len(edges_df) / (entity_num * entity_num)
        self.logger.info('Sparsity: {:.4f}'.format(sparsity))

        self.logger.info('\nCreating knowledge graph...')
        kg = self._create_kg(edges_df)
        self.logger.info('Data processed.\n')

        return {'edges': edges_df, 'kg': kg, 'label': label_train_df, 
                'user_id': user_id_train_list, 'target_entity_id': target_entity_id_train_list, 
                'userid2entityid': user2entity_dict, 'userid2relationship': user2relationship_dict,
                'target_entityid2entityid': target_entity2entity_dict, 'target_entityid2relationship': target_entity2relationship_dict,
                'node2id': node2id, 'id2node': id2node, 'relationship2id': relationship2id}
    
    def _create_label_validation(self, edges_valid_user_df, edges_train_df, 
                                 all_user_list, all_target_entity_list, node2id, relationship2id):        
        # Use multimodal data to get the same labels
        if not self.use_multimodal:
            edges_valid_user_df = edges_valid_user_df[(edges_valid_user_df['data_source'] != 'txt_jpg') | 
                                                      (edges_valid_user_df['relationship'] == self.label_relationship)]
        else:
            multimodal_edges_df = edges_valid_user_df[(edges_valid_user_df['data_source'] == 'txt_jpg') & 
                                                      (edges_valid_user_df['relationship'] != self.label_relationship)]
            edges_valid_user_df = edges_valid_user_df[(edges_valid_user_df['data_source'] != 'txt_jpg') | 
                                                      (edges_valid_user_df['relationship'] == self.label_relationship)]
        
        edges_no_label_df = edges_valid_user_df[edges_valid_user_df['relationship'] != self.label_relationship]
        if self.use_multimodal:
            edges_no_label_df = pd.concat([edges_no_label_df, multimodal_edges_df], ignore_index=True)
        edges_label_df = edges_valid_user_df[edges_valid_user_df['relationship'] == self.label_relationship]
        label_df = edges_label_df[['source', 'target', 'relationship']]
        self.logger.info('{} labels generated with unique {} source nodes and {} target nodes.'.format(
            label_df.shape, len(set(label_df['source'])), len(set(label_df['target']))))
        item_list = list(set(edges_label_df['source']) | set(edges_label_df['target']))
        user_valid_list = list(set(item_list) & set(all_user_list))
        target_entity_valid_list = list(set(item_list) & set(all_target_entity_list))
        entities_train_list = list(set(edges_train_df['source']) | set(edges_train_df['target']))
        relationships_train_list = list(set(edges_train_df['relationship']))
        
        # create user's entities, relationships for validation and test        
        user2entity_dict = {user: [] for user in user_valid_list}
        user2relationship_dict = {user: [] for user in user_valid_list}
        target_entity2entity_dict = {entity: [] for entity in target_entity_valid_list}
        target_entity2relationship_dict = {entity: [] for entity in target_entity_valid_list}
        for i, row in edges_no_label_df.iterrows():
            if row['relationship'] not in relationships_train_list:
                continue

            if row['source'] in user_valid_list:
                if row['target'] not in entities_train_list:
                    # cannot find any targets or relationships in training data
                    continue
                user2entity_dict[row['source']].append(row['target'])
                user2relationship_dict[row['source']].append(row['relationship'])
            if row['target'] in user_valid_list:
                if row['source'] not in entities_train_list:
                    # cannot find any sources or relationships in training data
                    continue
                user2entity_dict[row['target']].append(row['source'])
                user2relationship_dict[row['target']].append(row['relationship'])

            if row['source'] in target_entity_valid_list:
                if row['target'] not in entities_train_list:
                    # cannot find any sources or relationships in training data
                    continue
                target_entity2entity_dict[row['source']].append(row['target'])
                target_entity2relationship_dict[row['source']].append(row['relationship'])
            if row['target'] in target_entity_valid_list:
                if row['source'] not in entities_train_list:
                    # cannot find any sources or relationships in training data
                    continue
                target_entity2entity_dict[row['target']].append(row['source'])
                target_entity2relationship_dict[row['target']].append(row['relationship'])
        # remove users and entities that do not have valid relationships
        user2entity_dict = {user: value for user, value in user2entity_dict.items() if len(value) > 0}
        user2relationship_dict = {user: value for user, value in user2relationship_dict.items() if len(value) > 0}
        target_entity2entity_dict = {entity: value for entity, value in target_entity2entity_dict.items() if len(value) > 0}
        target_entity2relationship_dict = {entity: value for entity, value in target_entity2relationship_dict.items() if len(value) > 0}
        user_valid_list = list(user2entity_dict.keys())
        target_entity_valid_list = list(target_entity2entity_dict.keys())
        
        self.logger.info('Only keeping labels with valid users and target entitys...')
        label_organized_df_dict = {'user': [], 'target_entity': [], 'label': []}
        for i, row in label_df.iterrows():
            if row['source'] in user_valid_list and row['target'] in target_entity_valid_list:
                label_organized_df_dict['user'].append(row['source'])
                label_organized_df_dict['target_entity'].append(row['target'])
                label_organized_df_dict['label'].append(1)
            elif row['source'] in target_entity_valid_list and row['target'] in user_valid_list:
                label_organized_df_dict['user'].append(row['target'])
                label_organized_df_dict['target_entity'].append(row['source'])
                label_organized_df_dict['label'].append(1)
        label_df = pd.DataFrame(label_organized_df_dict)
        self.logger.info('{} labels generated.'.format(label_df.shape))

        # map
        valid_node2id, valid_id2node = {}, {}
        external_user_count, external_entity_count = 0, 0
        for user in list(set(label_df['user'])):
            if user in node2id:
                valid_node2id[user] = node2id[user]
            else:
                external_user_count += 1
                valid_node2id[user] = len(node2id) + external_user_count
                valid_id2node[valid_node2id[user]] = user
        self.logger.info('{} external users found.'.format(external_user_count))
        for entity in list(set(label_df['target_entity'])):
            if entity in node2id:
                valid_node2id[entity] = node2id[entity]
            else:
                external_entity_count += 1
                valid_node2id[entity] = len(node2id) + external_entity_count + external_user_count
                valid_id2node[valid_node2id[entity]] = entity
        self.logger.info('{} external target entity found.'.format(external_entity_count))
        label_df['user_id'] = label_df['user'].map(valid_node2id)
        label_df['target_entity_id'] = label_df['target_entity'].map(valid_node2id)

        # map entities and relationships in the kg
        kg_node2id = {**valid_node2id, **node2id}
        userid2entityid, userid2relationship = {}, {}
        target_entityid2entityid, target_entityid2relationship = {}, {}
        for user in list(user2entity_dict.keys()):
            if user not in kg_node2id:
                continue
            userid2entityid[kg_node2id[user]] = [kg_node2id[entity] for entity in user2entity_dict[user]]
            userid2relationship[kg_node2id[user]] = [relationship2id[relationship] for relationship in user2relationship_dict[user]]
        for target_entity in list(target_entity2entity_dict.keys()):
            if target_entity not in kg_node2id:
                continue
            target_entityid2entityid[kg_node2id[target_entity]] = [kg_node2id[user] for user in target_entity2entity_dict[target_entity]]
            target_entityid2relationship[kg_node2id[target_entity]] = [relationship2id[relationship] for relationship in target_entity2relationship_dict[target_entity]]

        return {'label': label_df, 'userid2entityid': userid2entityid, 'userid2relationship': userid2relationship,
                'target_entityid2entityid': target_entityid2entityid, 'target_entityid2relationship': target_entityid2relationship,
                'external_user_count': external_user_count, 'external_entity_count': external_entity_count}
    
    def _create_recommendation_label_validation(self, test_label_df, test_userid2entityid, test_userid2relationship,
                                                test_target_entityid2entityid, test_target_entityid2relationship,
                                                test_external_user_count, test_external_entity_count,
                                                node2id, target_entity_id_train_list, train_target_entityid2entityid, train_target_entityid2relationship):        

        # get negative samples across all target entities for recommendation
        neg_label_df_dict = {'user': [], 'user_id': [], 'target_entity': [], 'target_entity_id': [], 'label': []}
        id2node = {v: k for k, v in node2id.items()}
        target_entity_train_list = [id2node[v] for v in target_entity_id_train_list]
        test_user2id = dict(zip(list(test_label_df['user']), list(test_label_df['user_id'])))
        for user in list(set(test_label_df['user'])):
            user_target_entity_list = list(set(test_label_df[test_label_df['user'] == user]['target_entity']))
            for target_entity in list(set(target_entity_train_list) - set(user_target_entity_list)):
                neg_label_df_dict['user'].append(user)
                neg_label_df_dict['user_id'].append(test_user2id[user])
                neg_label_df_dict['target_entity'].append(target_entity)
                neg_label_df_dict['target_entity_id'].append(node2id[target_entity])
                neg_label_df_dict['label'].append(0)
        neg_label_df = pd.DataFrame(neg_label_df_dict)
        test_label_df = pd.concat([test_label_df, neg_label_df], ignore_index=True)
        self.logger.info('{} labels generated for recommendation.'.format(test_label_df.shape))
        # get target entity neighbors in the training data
        target_entityid2entityid= {**train_target_entityid2entityid, **test_target_entityid2entityid}
        target_entityid2relationship = {**train_target_entityid2relationship, **test_target_entityid2relationship}

        return {'label': test_label_df, 'userid2entityid': test_userid2entityid, 'userid2relationship': test_userid2relationship,
                'target_entityid2entityid': target_entityid2entityid, 'target_entityid2relationship': target_entityid2relationship,
                'external_user_count': test_external_user_count, 'external_entity_count': test_external_entity_count}
    
    def _process_data_kfold(self, K=5):
        prefix = ''
        if self.use_multimodal:
            prefix = 'multimodal-Seed{}-'.format(self.seed)
        else:
            prefix = 'no_multimodal-Seed{}-'.format(self.seed)
        prefix += self.label_relationship
        train_data_save_dir = self.save_dir.joinpath(prefix + '-data_train_dict-kfold0.pkl')
        
        if train_data_save_dir.exists():
            self.logger.info('Loading processed data...') 
        else:
            self.logger.info('\nProcessing data...')
            edges_df = self._load_edges()
            nodes_df = self._load_nodes()
            user_nodes_df, user_edges_df = self._load_users_as_edges()

            all_nodes_df = pd.concat([nodes_df, user_nodes_df], ignore_index=True)
            all_nodes_df = all_nodes_df.drop_duplicates(subset=['name'])
            all_user_list = list(set(all_nodes_df[all_nodes_df['attribute'] == 'username']['name']))
            all_target_entity_list = list(set(all_nodes_df[all_nodes_df['attribute'] == self.target_entity_name]['name']))

            # split users to train, valid, and test
            user_index_list = list(range(len(user_nodes_df)))
            np.random.seed(self.seed)
            np.random.shuffle(user_index_list)
            fold_size = len(user_index_list) // K
            fold_index_list = [user_index_list[i*fold_size: (i+1)*fold_size] for i in range(K)]
            if len(fold_index_list) % K != 0:
                fold_index_list[-1] += user_index_list[K*fold_size:]

            for fold_idx in range(K):
                if fold_idx > 0:
                    continue
                self.logger.info('Processing fold {}...'.format(fold_idx))
                test_index_list = fold_index_list[fold_idx]
                train_index_list = []
                for i in range(K):
                    if i != fold_idx:
                        train_index_list += fold_index_list[i]
                # 10% of training data as validation set
                np.random.shuffle(train_index_list)
                valid_index_list = train_index_list[: int(len(train_index_list)*0.2)]
                train_index_list = train_index_list[int(len(train_index_list)*0.2): ]

                user_nodes_train_df = user_nodes_df.iloc[train_index_list]
                user_nodes_valid_df = user_nodes_df.iloc[valid_index_list]
                user_nodes_test_df = user_nodes_df.iloc[test_index_list]

                user_train_list = list(set(user_nodes_train_df['name']))
                user_valid_list = list(set(user_nodes_valid_df['name']))
                user_test_list = list(set(user_nodes_test_df['name']))
                user_df = pd.DataFrame({'username': user_train_list + user_valid_list + user_test_list,
                                        'split': ['train']*len(user_train_list) + ['valid']*len(user_valid_list) + ['test']*len(user_test_list)})
                user_df.to_csv(self.save_dir.joinpath(prefix + 'users_split-kfold{}.csv'.format(fold_idx)), index=False)

                user_edges_train_df = user_edges_df[user_edges_df['source_user'].isin(user_train_list)]
                user_edges_valid_df = user_edges_df[user_edges_df['source_user'].isin(user_valid_list)]
                user_edges_test_df = user_edges_df[user_edges_df['source_user'].isin(user_test_list)]

                self.logger.info('Preparing data for training...')
                nodes_train_df = nodes_df[~nodes_df['source_user'].isin(user_test_list)]
                edges_train_df = edges_df[~edges_df['source_user'].isin(user_test_list)]
                data_train_dict = self._create_kg_label_train(nodes_train_df, edges_train_df, user_nodes_train_df, user_edges_train_df)
                edges_train_df = data_train_dict['edges']

                self.logger.info('Preparing data for validation...')
                edges_valid_df = edges_df[edges_df['source_user'].isin(user_valid_list)]
                edges_valid_user_df = pd.concat([user_edges_valid_df, edges_valid_df], ignore_index=True)
                edges_valid_user_df = edges_valid_user_df.drop_duplicates(subset=['source', 'target', 'relationship'])
                data_valid_dict = self._create_label_validation(edges_valid_user_df, edges_train_df, all_user_list, all_target_entity_list,
                                                                data_train_dict['node2id'], data_train_dict['relationship2id'])
                
                self.logger.info('Preparing data for test...')
                edges_test_df = edges_df[edges_df['source_user'].isin(user_test_list)]
                edges_test_user_df = pd.concat([user_edges_test_df, edges_test_df], ignore_index=True)
                edges_test_user_df = edges_test_user_df.drop_duplicates(subset=['source', 'target', 'relationship'])
                data_test_dict = self._create_label_validation(edges_test_user_df, edges_train_df, all_user_list, all_target_entity_list,
                                                               data_train_dict['node2id'], data_train_dict['relationship2id'])

                self.logger.info('Saving data...')
                with open(self.save_dir.joinpath(prefix + '-data_train_dict-kfold{}.pkl'.format(fold_idx)), 'wb') as f:
                    pickle.dump(data_train_dict, f)
                with open(self.save_dir.joinpath(prefix + '-data_valid_dict-kfold{}.pkl'.format(fold_idx)), 'wb') as f:
                    pickle.dump(data_valid_dict, f)
                with open(self.save_dir.joinpath(prefix + '-data_test_dict-kfold{}.pkl'.format(fold_idx)), 'wb') as f:
                    pickle.dump(data_test_dict, f)

    def _negative_sampling(self, label_df, target_entity_id_list, id2node):
        self.logger.info('Negative sampling for each user...')
        neg_label_df_dict = {'user_id': [], 'target_entity_id': [], 'label': []}
        for user in list(set(label_df['user_id'])):
            user_label_df = label_df[label_df['user_id'] == user]
            neg_product_category_list = list(set(target_entity_id_list) - set(user_label_df['target_entity_id']))
            neg_product_category_list = random.sample(neg_product_category_list, len(user_label_df))
            # neg_product_category_list = list(np.random.choice(neg_product_category_list, size=len(user_label_df), replace=True))
            neg_label_df_dict['user_id'] += [user] * len(neg_product_category_list)
            neg_label_df_dict['target_entity_id'] += neg_product_category_list
            neg_label_df_dict['label'] += [0] * len(neg_product_category_list)
        neg_label_df = pd.DataFrame(neg_label_df_dict)
        neg_label_df['user'] = neg_label_df['user_id'].map(id2node)
        neg_label_df['target_entity'] = neg_label_df['target_entity_id'].map(id2node)
        self.logger.info('{} negative samples generated.'.format(neg_label_df.shape))
        return neg_label_df
    
    def get_dataloader_kfold(self, K_idx):
        self.logger.info('Creating dataloader for {}-fold...'.format(K_idx))
        prefix = ''
        if self.use_multimodal:
            prefix = 'multimodal-Seed{}-'.format(self.seed)
        else:
            prefix = 'no_multimodal-Seed{}-'.format(self.seed)
        prefix += self.label_relationship

        with open(self.save_dir.joinpath(prefix + '-data_train_dict-kfold{}.pkl'.format(K_idx)), 'rb') as f:
            data_train_dict = pickle.load(f)
        with open(self.save_dir.joinpath(prefix + '-data_valid_dict-kfold{}.pkl'.format(K_idx)), 'rb') as f:
            data_valid_dict = pickle.load(f)
        with open(self.save_dir.joinpath(prefix + '-data_test_dict-kfold{}.pkl'.format(K_idx)), 'rb') as f:
            data_test_dict = pickle.load(f)
        
        train_dataset = KGDataset(data_train_dict, self.n_neighbor)
        valid_dataset = KGDataset(data_valid_dict, self.n_neighbor)
        test_dataset = KGDataset(data_test_dict, self.n_neighbor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        return data_train_dict, train_loader, valid_loader, test_loader
    
    def get_dataloader_inference(self):
        self.logger.info('Creating dataloader recommendation...')
        prefix = ''
        if self.use_multimodal:
            prefix = 'multimodal-Seed{}-'.format(self.seed)
        else:
            prefix = 'no_multimodal-Seed{}-'.format(self.seed)
        prefix += self.label_relationship

        K_idx = 0
        with open(self.save_dir.joinpath(prefix + '-data_train_dict-kfold{}.pkl'.format(K_idx)), 'rb') as f:
            data_train_dict = pickle.load(f)
        with open(self.save_dir.joinpath(prefix + '-data_valid_dict-kfold{}.pkl'.format(K_idx)), 'rb') as f:
            data_valid_dict = pickle.load(f)
        with open(self.save_dir.joinpath(prefix + '-data_test_dict-kfold{}.pkl'.format(K_idx)), 'rb') as f:
            data_test_dict = pickle.load(f)

        if not self.save_dir.joinpath(prefix + '-data_recommendation_test_dict-kfold{}.pkl'.format(K_idx)).exists():
            data_test_recommendation_dict = self._create_recommendation_label_validation(
                data_test_dict['label'], data_test_dict['userid2entityid'], data_test_dict['userid2relationship'],
                data_test_dict['target_entityid2entityid'], data_test_dict['target_entityid2relationship'],
                data_test_dict['external_user_count'], data_test_dict['external_entity_count'],
                data_train_dict['node2id'], data_train_dict['target_entity_id'], 
                data_train_dict['target_entityid2entityid'], data_train_dict['target_entityid2relationship'])
            with open(self.save_dir.joinpath(prefix + '-data_recommendation_test_dict-kfold{}.pkl'.format(K_idx)), 'wb') as f:
                pickle.dump(data_test_recommendation_dict, f)
        else:
            with open(self.save_dir.joinpath(prefix + '-data_recommendation_test_dict-kfold{}.pkl'.format(K_idx)), 'rb') as f:
                data_test_recommendation_dict = pickle.load(f)

        train_dataset = KGDataset(data_train_dict, self.n_neighbor)   
        valid_dataset = KGDataset(data_valid_dict, self.n_neighbor)     
        test_recommendation_dataset = KGDataset(data_test_recommendation_dict, self.n_neighbor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        test_recommendation_loader = DataLoader(test_recommendation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        return data_train_dict, train_loader, valid_loader, test_recommendation_loader
    
    def get_dataloader_node2vec(self):
        self.logger.info('Creating dataloader for node2vec...')
        label_df = pd.concat([self.label_df, self.neg_label_df], ignore_index=True)
        label_df = label_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        self.logger.info('label distribution: {}'.format(label_df['label'].value_counts()))

        train_dataset = KGDataset(label_df)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        return train_loader