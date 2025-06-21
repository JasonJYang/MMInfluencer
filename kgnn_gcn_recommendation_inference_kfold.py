import argparse
import collections
import torch
import numpy as np
import pandas as pd
import model.metric as module_metric
import data_loader.data_loaders as module_data
from tqdm import tqdm
from parse_config import ConfigParser


def kgnn_gcn_inference(config, SEED, logger):
    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)
    kg = data_loader.get_kg()
    entity_num = data_loader.get_entity_num()
    relation_num = data_loader.get_relation_num()
    node2id = data_loader.get_node2id()
    id2node = data_loader.get_id2node()
    entity_num = entity_num + 1  # add 1 for padding entity
    relation_num = relation_num + 1  # add 1 for padding relation
    logger.info(f'entity_num: {entity_num}, relation_num: {relation_num} with padding entity/relation')
    dataloader_list = data_loader.get_dataloader_kfold(config['k_fold'])

    for i in range(config['k_fold']):
        logger.info('------------------------------------------------')
        logger.info('------------------Fold {}-----------------------'.format(i))
        logger.info('------------------------------------------------')
        train_data_loader, valid_data_loader, test_data_loader = dataloader_list[i]
        inference_data_loader = data_loader.get_dataloader_inference_kth_fold(
            k=i, resume_dir=config['resume'], node2id=node2id, id2node=id2node)

        logger.info('Loading model...')
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
            from model.gcn_model import GCN
            model = GCN(entity_num=entity_num,
                        kg=kg,
                        emb_dim=config['arch']['args']['emb_dim'],
                        layersize=config['arch']['args']['layersize'],
                        dropout=config['arch']['args']['dropout'])
        
        if i == 0:
            logger.info(model)

        metrics = [getattr(module_metric, met) for met in config['metrics']]

        from trainer.gcn_trainer import Trainer  
        trainer = Trainer(model, None, metrics, None,
                          config=config,
                          train_data_loader=train_data_loader,
                          valid_data_loader=valid_data_loader,
                          test_data_loader=inference_data_loader,
                          lr_scheduler=None)
        if config['arch']['type'] == 'KGNN':
            trainer.model.device = trainer.device
            trainer.model.adj_ent = model.adj_ent.to(trainer.device)
            trainer.model.adj_rel = model.adj_rel.to(trainer.device)
        trainer.current_k = i

        """Inference."""
        logger = config.get_logger('test')
        logger.info(model)
        
        # load best checkpoint
        resume = str(config['resume']) + '/model_best_K{}.pth'.format(i)
        logger.info('Loading checkpoint: {} ...'.format(resume))
        checkpoint = torch.load(resume)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        test_result_df = trainer.recommendation_inference()
        test_result_df.to_csv(str(config.save_dir / 'test_result_K{}.csv'.format(i)), index=False)
        logger.info('------------------Finish testing----------------------')

    logger.info('--------------------Finish K-fold!-------------------------')


def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    SEED = config['data_loader']['args']['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    if config['arch']['type'] == 'KGNN' or config['arch']['type'] == 'GCN':
        kgnn_gcn_inference(config, SEED, logger)    


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
