import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders_zero_shot as module_data
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('train')

    for K in range(5):
        # fix random seeds for reproducibility
        SEED = config['seed'] + K
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)

        # setup data_loader instances
        config['data_loader']['args']['logger'] = logger
        data_loader = config.init_obj('data_loader', module_data)
        data_train_dict, train_data_loader, valid_data_loader, test_data_loader = data_loader.get_dataloader_kfold(K_idx=K)
        kg = data_loader.get_kg(data_train_dict)
        entity_num = data_loader.get_entity_num(data_train_dict)
        relation_num = data_loader.get_relation_num(data_train_dict)
        entity_num = entity_num + 1  # add 1 for padding entity
        relation_num = relation_num + 1  # add 1 for padding relation
        logger.info(f'entity_num: {entity_num}, relation_num: {relation_num} with padding entity/relation')

        # build model architecture, then print to console
        if config['arch']['type'] == 'KGNN':
            from model.kgnn_model import KGNN
            model = KGNN(entity_num=entity_num,
                        relation_num=relation_num,
                        emb_dim=config['arch']['args']['emb_dim'],
                        kg=kg,
                        seed=SEED,
                        n_hop=config['arch']['args']['n_hop'],
                        n_neighbor=config['arch']['args']['n_neighbor'],
                        dropout=config['arch']['args']['dropout'],
                        aggregator_name=config['arch']['args']['aggregator_name'])
        
        elif config['arch']['type'] == 'GCN':
            from model.gcn_model import GCNInductive
            model = GCNInductive(entity_num=entity_num,
                                 kg=kg,
                                 emb_dim=config['arch']['args']['emb_dim'],
                                 layersize=config['arch']['args']['layersize'],
                                 dropout=config['arch']['args']['dropout'])
        
        elif config['arch']['type'] == 'Node2Vec':
            from model.node2vec_model import Node2VecInductiveModel
            model = Node2VecInductiveModel(entity_num=entity_num,
                                           kg=kg,
                                           emb_dim=config['arch']['args']['emb_dim'],
                                           walk_length=config['arch']['args']['walk_length'],
                                           context_size=config['arch']['args']['context_size'],
                                           walks_per_node=config['arch']['args']['walks_per_node'],
                                           num_negative_samples=config['arch']['args']['num_negative_samples'])

        elif config['arch']['type'] == 'TransE':
            from model.transe_model import TransEInductiveModel
            model = TransEInductiveModel(entity_num=entity_num,
                                         relation_num=relation_num,
                                         kg=kg,
                                         emb_dim=config['arch']['args']['emb_dim'])

        logger.info(model)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        if config['arch']['type'] == 'KGNN' or config['arch']['type'] == 'GCN':
            from trainer.gcn_trainer_zero_shot import Trainer
        
        elif config['arch']['type'] == 'Node2Vec':
            from trainer.node2vec_trainer import Trainer
            train_data_loader = model.node2vec.loader(batch_size=config['data_loader']['args']['batch_size'], 
                                                      shuffle=config['data_loader']['args']['shuffle'], 
                                                      num_workers=config['data_loader']['args']['num_workers'])
            
        elif config['arch']['type'] == 'TransE':
            from trainer.transe_trainer import Trainer
            edge_index, edge_type = model._get_kg_index()
            train_data_loader = model.transe.loader(head_index=edge_index[0],
                                                    rel_type=edge_type,
                                                    tail_index=edge_index[1],
                                                    batch_size=config['data_loader']['args']['batch_size'], 
                                                    shuffle=config['data_loader']['args']['shuffle'], 
                                                    num_workers=config['data_loader']['args']['num_workers'])
        
        trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        train_data_loader=train_data_loader,
                        valid_data_loader=valid_data_loader,
                        test_data_loader=test_data_loader,
                        lr_scheduler=lr_scheduler)
        
        if config['arch']['type'] == 'KGNN':
            trainer.model.device = trainer.device
            trainer.model.adj_ent = model.adj_ent.to(trainer.device)
            trainer.model.adj_rel = model.adj_rel.to(trainer.device)

        trainer.train()

        """Test."""
        logger = config.get_logger('test')
        logger.info(model)
        
        # load best checkpoint
        if config['arch']['type'] == 'KGNN' or config['arch']['type'] == 'GCN':
            resume = str(config.save_dir / 'model_best.pth')
            logger.info('Loading checkpoint: {} ...'.format(resume))
            checkpoint = torch.load(resume)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)

        if config['arch']['type'] == 'Node2Vec' or config['arch']['type'] == 'TransE':
            test_output, test_result_df = trainer.test_coldstart()
        else:
            test_output, test_result_df = trainer.test()
        value_format = ''.join(['{:15s}: {:.2f}\t'.format(k, v) for k, v in test_output.items()])
        logger.info('    {:15s}: {}'.format('test', value_format))
        test_result_df.to_csv(str(config.save_dir / 'test_result-K{}.csv'.format(K)), index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
