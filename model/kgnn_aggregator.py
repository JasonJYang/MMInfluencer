import torch
import torch.nn.functional as F


# source: https://github.com/zzaebok/KGCN-pytorch/blob/master/aggregator.py
class Aggregator(torch.nn.Module):
    '''
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    '''
    def __init__(self, dim, aggregator_name):
        super(Aggregator, self).__init__()
        self.dim = dim
        if aggregator_name == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator_name = aggregator_name
        
    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, target_embeddings, act):
        batch_size = target_embeddings.size(0)
        neighbors_agg = self._mix_neighbor_vectors(batch_size, neighbor_vectors, neighbor_relations, target_embeddings)
        
        if self.aggregator_name == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))
            
        elif self.aggregator_name == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))
            
        else:
            output = neighbors_agg.view((-1, self.dim))
            
        output = self.weights(output)
        return act(output.view((batch_size, -1, self.dim)))
        
    def _mix_neighbor_vectors(self, batch_size, neighbor_vectors, neighbor_relations, user_embeddings):
        '''
        This aims to aggregate neighbor vectors
        '''
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((batch_size, 1, 1, self.dim))
        
        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim = -1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim = -1)
        
        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim = -1)
        
        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim = 2)
        
        return neighbors_aggregated