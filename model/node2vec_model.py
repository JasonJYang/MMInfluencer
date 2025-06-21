import torch
import torch.nn as nn
from torch_geometric.nn import Node2Vec
from base import BaseModel

class Node2VecModel(BaseModel):
    def __init__(self, entity_num, 
                       kg,
                       emb_dim=16, 
                       walk_length=20,
                       context_size=10,
                       walks_per_node=10,
                       p=1,
                       q=1,
                       num_negative_samples=1):
        super(Node2VecModel, self).__init__()
        
        self.entity_num = entity_num
        self.edge_index = self._get_edge_index(kg)
        
        self.node2vec = Node2Vec(
            edge_index=self.edge_index,
            num_nodes=entity_num,
            embedding_dim=emb_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
            num_negative_samples=num_negative_samples,
            sparse=True)

    def _get_edge_index(self, kg):
        # Same as in GCN model
        edge_index = []
        for head in kg:
            for (rel, tail) in kg[head]:
                edge_index.append([head, tail])
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        return edge_index

    def forward(self, users, users_entities, users_relations, 
                      products, products_entities, products_relations):
        # Get embeddings for all nodes
        ego_embedding = self.node2vec(torch.arange(self.entity_num, device=users.device))

        # Get user and product embeddings
        user_embeddings = ego_embedding[users]
        product_embeddings = ego_embedding[products]
        
        # Compute similarity score
        score = torch.sum(user_embeddings * product_embeddings, dim=1)
        return score
    

class Node2VecInductiveModel(BaseModel):
    def __init__(self, entity_num, 
                       kg,
                       emb_dim=16, 
                       walk_length=20,
                       context_size=10,
                       walks_per_node=10,
                       p=1,
                       q=1,
                       num_negative_samples=1):
        super(Node2VecInductiveModel, self).__init__()
        
        self.entity_num = entity_num
        self.edge_index = self._get_edge_index(kg)
        
        self.node2vec = Node2Vec(
            edge_index=self.edge_index,
            num_nodes=entity_num,
            embedding_dim=emb_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            p=p,
            q=q,
            num_negative_samples=num_negative_samples,
            sparse=True)

    def _get_edge_index(self, kg):
        # Same as in GCN model
        edge_index = []
        for head in kg:
            for (rel, tail) in kg[head]:
                edge_index.append([head, tail])
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        return edge_index

    def forward(self, users, users_entities, users_relations, 
                      products, products_entities, products_relations):
        # Get embeddings for all nodes
        ego_embedding = self.node2vec(torch.arange(self.entity_num, device=users.device))

        users_entities_embeddings = ego_embedding[users_entities]
        users_embeddings = torch.sum(users_entities_embeddings, dim=1)

        products_entities_embeddings = ego_embedding[products_entities]
        products_embeddings = torch.sum(products_entities_embeddings, dim=1)

        score = torch.sum(users_embeddings * products_embeddings, dim=1)
        return score