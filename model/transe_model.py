import torch
from torch_geometric.nn import TransE
from base import BaseModel

# source: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/kge_fb15k_237.py
class TransEModel(BaseModel):
    def __init__(self, entity_num, 
                       relation_num,
                       kg, 
                       emb_dim=50):
        super(TransEModel, self).__init__()
        
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.kg = kg
        
        self.transe = TransE(
            num_nodes=entity_num,
            num_relations=relation_num,
            hidden_channels=emb_dim)
        
    def _get_kg_index(self):
        # Same as in GCN model
        edge_index_head, edge_index_tail = [], []
        edge_type = []
        for head in self.kg:
            for (rel, tail) in self.kg[head]:
                edge_index_head.append(head)
                edge_index_tail.append(tail)
                edge_type.append(rel)
        edge_index_head = torch.LongTensor(edge_index_head)
        edge_index_tail = torch.LongTensor(edge_index_tail)
        edge_index = torch.stack([edge_index_head, edge_index_tail], dim=0)
        edge_type = torch.LongTensor(edge_type)
        return edge_index, edge_type

    def forward(self, users, users_entities, users_relations, 
                      products, products_entities, products_relations):
        # Get embeddings for users and products
        user_embeddings = self.transe.node_emb(users)
        product_embeddings = self.transe.node_emb(products)
        
        # Compute similarity score
        score = torch.sum(user_embeddings * product_embeddings, dim=1)
        return score

class TransEInductiveModel(BaseModel):
    def __init__(self, entity_num, 
                       relation_num,
                       kg, 
                       emb_dim=50):
        super(TransEInductiveModel, self).__init__()
        
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.kg = kg
        
        self.transe = TransE(
            num_nodes=entity_num,
            num_relations=relation_num,
            hidden_channels=emb_dim)
        
    def _get_kg_index(self):
        # Same as in GCN model
        edge_index_head, edge_index_tail = [], []
        edge_type = []
        for head in self.kg:
            for (rel, tail) in self.kg[head]:
                edge_index_head.append(head)
                edge_index_tail.append(tail)
                edge_type.append(rel)
        edge_index_head = torch.LongTensor(edge_index_head)
        edge_index_tail = torch.LongTensor(edge_index_tail)
        edge_index = torch.stack([edge_index_head, edge_index_tail], dim=0)
        edge_type = torch.LongTensor(edge_type)
        return edge_index, edge_type

    def forward(self, users, users_entities, users_relations, 
                      products, products_entities, products_relations):
        # Get embeddings for users and products
        users_entities_embeddings = self.transe.node_emb(users_entities)
        users_embeddings = torch.sum(users_entities_embeddings, dim=1)

        products_entities_embeddings = self.transe.node_emb(products_entities)
        products_embeddings = torch.sum(products_entities_embeddings, dim=1)
        
        # Compute similarity score
        score = torch.sum(users_embeddings * products_embeddings, dim=1)
        return score