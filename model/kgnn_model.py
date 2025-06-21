import torch
import random
import torch.nn as nn
from base import BaseModel
from model.kgnn_aggregator import Aggregator


class KGNN(BaseModel):
    def __init__(self, entity_num, 
                       relation_num,
                       emb_dim, 
                       kg,
                       seed,
                       n_hop,
                       n_neighbor,
                       dropout=0.5,
                       aggregator_name='sum'):
        super(KGNN, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.emb_dim = emb_dim
        self.kg = kg
        random.seed(seed)
        self.n_neighbor = n_neighbor
        self.n_hop = n_hop
        self.dropout = dropout

        self.adj_ent, self.adj_rel = self._gen_adj()

        self.entity_embedding_table = nn.Embedding(self.entity_num, self.emb_dim)
        self.relation_embedding_table = nn.Embedding(self.relation_num, self.emb_dim)
        self.aggregator = Aggregator(dim=self.emb_dim, aggregator_name=aggregator_name)

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        adj_ent = torch.empty(self.entity_num, self.n_neighbor, dtype=torch.long)
        adj_rel = torch.empty(self.entity_num, self.n_neighbor, dtype=torch.long)
        
        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
                
            adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])

        return adj_ent, adj_rel

    def forward(self, users, users_entities, users_relations, 
                      products, products_entities, products_relations):
        batch_size = users.size(0)
        user_embeddings = self.entity_embedding_table(users)
        product_embeddings = self.entity_embedding_table(products)

        users = users.view((-1, 1))
        products = products.view((-1, 1))

        # get user embeddings using kg
        entities, relations = self._get_neighbors(batch_size, users)
        user_embeddings = self._interaction_aggregation(batch_size, entities, relations, product_embeddings)

        # get product embeddings using kg
        entities, relations = self._get_neighbors(batch_size, products)
        product_embeddings = self._interaction_aggregation(batch_size, entities, relations, user_embeddings)

        score = torch.sum(user_embeddings * product_embeddings, dim=1)
        return score
    
    def _get_neighbors(self, batch_size, seeds):
        entities = [seeds]
        relations = []
        for h in range(self.n_hop):
            neighbor_entities = self.adj_ent[entities[h]].view((batch_size, -1))
            neighbor_relations = self.adj_rel[entities[h]].view((batch_size, -1))
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def _interaction_aggregation(self, batch_size, entities, relations, target_embeddings):
        entity_embeddings = [self.entity_embedding_table(i) for i in entities]
        relation_embeddings = [self.relation_embedding_table(i) for i in relations]

        for i in range(self.n_hop):
            if i == self.n_hop - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid
            
            entity_embeddings_next_hop = []
            for hop in range(self.n_hop - i):
                embedding = self.aggregator(self_vectors=entity_embeddings[hop],
                                            neighbor_vectors=entity_embeddings[hop + 1].view((batch_size, -1, self.n_neighbor, self.emb_dim)),
                                            neighbor_relations=relation_embeddings[hop].view((batch_size, -1, self.n_neighbor, self.emb_dim)),
                                            target_embeddings=target_embeddings,
                                            act=act)
                entity_embeddings_next_hop.append(embedding)
            entity_embeddings = entity_embeddings_next_hop

        res = entity_embeddings[0].view((batch_size, self.emb_dim))
        return res

    def _l2_loss(self, device):
        l2_loss = torch.tensor(0.0).to(device)
        for param in self.parameters():
            l2_loss += torch.norm(param)
        return l2_loss

    def inference_zero_shot(self, users, users_entities, users_relations, 
                                   products, products_entities, products_relations):
        def _get_neighbors(batch_size, entity_seeds, entity_neighbors, entity_relations):
            entities = [entity_seeds.view((batch_size, -1))]
            relations = []
            # first hop
            entities.append(entity_neighbors.view((batch_size, -1)))
            relations.append(entity_relations.view((batch_size, -1)))

            for h in range(1, self.n_hop):
                neighbor_entities = self.adj_ent[entities[h]].view((batch_size, -1))
                neighbor_relations = self.adj_rel[entities[h]].view((batch_size, -1))
                entities.append(neighbor_entities)
                relations.append(neighbor_relations)
            return entities, relations
        
        def _interaction_aggregation(batch_size, entities, relations, original_embeddings, target_embeddings):
            entity_embeddings = [original_embeddings] + [self.entity_embedding_table(v) for idx, v in enumerate(entities) if idx > 0]
            relation_embeddings = [self.relation_embedding_table(v) for idx, v in enumerate(relations)]

            for i in range(self.n_hop):
                if i == self.n_hop - 1:
                    act = torch.tanh
                else:
                    act = torch.sigmoid
                
                entity_embeddings_next_hop = []
                for hop in range(self.n_hop - i):
                    embedding = self.aggregator(self_vectors=entity_embeddings[hop],
                                                neighbor_vectors=entity_embeddings[hop + 1].view((batch_size, -1, self.n_neighbor, self.emb_dim)),
                                                neighbor_relations=relation_embeddings[hop].view((batch_size, -1, self.n_neighbor, self.emb_dim)),
                                                target_embeddings=target_embeddings,
                                                act=act)
                    entity_embeddings_next_hop.append(embedding)
                entity_embeddings = entity_embeddings_next_hop

            res = entity_embeddings[0].view((batch_size, self.emb_dim))
            return res
        
        batch_size = users.size(0)
        # create a random embedding if user or product is not in the training set
        user_embedding_list = []
        for user in users:
            if user.item() >= self.entity_num:
                user_embedding_list.append(torch.randn(self.emb_dim).to(users.device))
            else:
                user_embedding_list.append(self.entity_embedding_table(user))
        product_embedding_list = []
        for product in products:
            if product.item() >= self.entity_num:
                product_embedding_list.append(torch.randn(self.emb_dim).to(products.device))
            else:
                product_embedding_list.append(self.entity_embedding_table(product))
        
        user_embeddings = torch.stack(user_embedding_list, dim=0).to(users.device)
        user_embeddings = torch.unsqueeze(user_embeddings, dim=1)
        product_embeddings = torch.stack(product_embedding_list, dim=0).to(products.device)
        product_embeddings = torch.unsqueeze(product_embeddings, dim=1)

        # get user embeddings using kg
        entities, relations = _get_neighbors(batch_size, users, users_entities, users_relations)
        user_embeddings = _interaction_aggregation(batch_size, entities, relations, user_embeddings, product_embeddings)

        # for products, we first find whether the product is in the training set
        entities, relations = _get_neighbors(batch_size, products, products_entities, products_relations)
        product_embeddings = _interaction_aggregation(batch_size, entities, relations, product_embeddings, user_embeddings)

        score = torch.sum(user_embeddings * product_embeddings, dim=1)
        return score