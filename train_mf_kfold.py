import argparse
import collections
import torch
import numpy as np
import data_loader.mf_data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    SEED = config['data_loader']['args']['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)
    user_num = data_loader.get_user_num()
    entity_num = data_loader.get_entity_num()
    logger.info(f'user_num: {user_num}, entity_num: {entity_num}') 
    dataloader_list = data_loader.get_dataloader_kfold(config['k_fold'])

    logger.info('------------------Start training-----------------------')
    logger.info('K-fold: {}'.format(config['k_fold']))

    for i in range(config['k_fold']):
        logger.info('------------------------------------------------')
        logger.info('------------------Fold {}-----------------------'.format(i))
        logger.info('------------------------------------------------')
        train_data_loader, valid_data_loader, test_data_loader = dataloader_list[i]

        logger.info('Loading model...')
        # build model architecture, then print to console
        if config['arch']['type'] == 'MatrixFactorization':
            from model.mf_model import MatrixFactorization
            model = MatrixFactorization(num_users=user_num,
                                        num_items=entity_num,
                                        emb_dim=config['arch']['args']['emb_dim'],
                                        reg_lambda=config['arch']['args']['reg_lambda']) 
        if i == 0:
            logger.info(model)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        if config['arch']['type'] == 'MatrixFactorization':
            from trainer.mf_trainer import MFTrainer as Trainer
        
        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          train_data_loader=train_data_loader,
                          valid_data_loader=valid_data_loader,
                          test_data_loader=test_data_loader,
                          lr_scheduler=lr_scheduler)
        
        trainer.current_k = i
        trainer.train()

        """Test."""
        logger = config.get_logger('test')
        logger.info(model)
        
        # load best checkpoint
        resume = str(config.save_dir / 'model_best_K{}.pth'.format(i))
        logger.info('Loading checkpoint: {} ...'.format(resume))
        checkpoint = torch.load(resume)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)

        test_output, test_result_df = trainer.test()
        test_result_df.to_csv(str(config.save_dir / 'test_result_K{}.csv'.format(i)), index=False)
        value_format = ''.join(['{:15s}: {:.2f}\t'.format(k, v) for k, v in test_output.items()])
        logger.info('    {:15s}: {}'.format('test', value_format))

        logger.info('------------------Finish testing----------------------')

    logger.info('--------------------Finish K-fold!-------------------------')


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
