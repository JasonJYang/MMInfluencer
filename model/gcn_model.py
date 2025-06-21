import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from base import BaseModel


class GCN(BaseModel):
    def __init__(self, entity_num, kg, emb_dim=16, layersize=[16, 16, 16], dropout=0):
        super(GCN, self).__init__()
        
        self.entity_num = entity_num
        self.kg = kg
        self.edge_index = self._get_edge_index(kg)
        
        gcns = []
        for layer in range(len(layersize)-1):
            gcns.append(GCNConv(in_channels=layersize[layer],
                                out_channels=layersize[layer+1],
                                cached=True,
                                add_self_loops=True))
        self.gcns = nn.ModuleList(gcns)
        
        self.embedding_table = nn.Embedding(entity_num, emb_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _get_edge_index(self, kg):
        edge_index = []
        for head in kg:
            for (rel, tail) in kg[head]:
                edge_index.append([head, tail])
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        return edge_index

    def forward(self, users, users_entities, users_relations, 
                      products, products_entities, products_relations):
        total_node = torch.LongTensor([list(range(self.entity_num))]).to(users.device)
        ego_embedding = self.embedding_table(total_node).squeeze()

        for gcn in self.gcns:
            ego_embedding = gcn(ego_embedding, self.edge_index.to(users.device))
            ego_embedding = self.activation(ego_embedding)
            ego_embedding = self.dropout(ego_embedding)

        user_embeddings = ego_embedding[users]
        product_embeddings = ego_embedding[products]
        
        score = torch.sum(user_embeddings * product_embeddings, dim=1)
        return score
    

class GCNInductive(BaseModel):
    def __init__(self, entity_num, kg, emb_dim=16, layersize=[16, 16, 16], dropout=0):
        super(GCNInductive, self).__init__()
        
        self.entity_num = entity_num
        self.kg = kg
        self.edge_index = self._get_edge_index(kg)
        
        gcns = []
        for layer in range(len(layersize)-1):
            gcns.append(GCNConv(in_channels=layersize[layer],
                                out_channels=layersize[layer+1],
                                cached=True,
                                add_self_loops=True))
        self.gcns = nn.ModuleList(gcns)
        
        self.embedding_table = nn.Embedding(entity_num, emb_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _get_edge_index(self, kg):
        edge_index = []
        for head in kg:
            for (rel, tail) in kg[head]:
                edge_index.append([head, tail])
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        return edge_index

    def forward(self, users, users_entities, users_relations, 
                      products, products_entities, products_relations):
        total_node = torch.LongTensor([list(range(self.entity_num))]).to(users.device)
        ego_embedding = self.embedding_table(total_node).squeeze()

        for gcn in self.gcns:
            ego_embedding = gcn(ego_embedding, self.edge_index.to(users.device))
            ego_embedding = self.activation(ego_embedding)
            ego_embedding = self.dropout(ego_embedding)

        users_entities_embeddings = ego_embedding[users_entities]
        users_embeddings = torch.sum(users_entities_embeddings, dim=1)

        products_entities_embeddings = ego_embedding[products_entities]
        products_embeddings = torch.sum(products_entities_embeddings, dim=1)

        score = torch.sum(users_embeddings * products_embeddings, dim=1)
        return score
    
    def inference_zero_shot(self, users, users_entities, users_relations, 
                                   products, products_entities, products_relations):
        return self.forward(users, users_entities, users_relations, products, products_entities, products_relations)
