import argparse

from pathlib import Path

# from src.make_folders import make_folders
from src.dataset import CBCTDataset
from src.dataprep import save_pickle
from helper_functions.helpers import load_config
from helper_functions.training_loggers import plot_loss
from exec_scripts.pretrain import pretrain
from preprocess.make_data import preprocess
from src.model_utils import embed_databank


def execute():
    # 1 Create parser
    parser = argparse.ArgumentParser(description='Main launcher for CBCT project.')
    subparsers = parser.add_subparsers(dest='command', help='Main command')
    subparsers.required = True

    prep_parser = subparsers.add_parser('prep', help='Run the data preparation pipeline')
    prep_parser.add_argument(
        '-t',
        '--type',
        choices=['full', 'sample'],
        default='full',
        help='The specific dataprep type to run.'
    )
    prep_parser.add_argument(
        '-cp',
        '--class_prop',
        type=float,
        default=0.5,
        help='Proportion of CT (vs CBCT) class.'
    )
    prep_parser.add_argument(
        '-sp',
        '--sample_prop',
        type=float,
        default=0.3,
        help='Proportion of data sample.'
    )
    prep_parser.add_argument(
        '-n',
        '--names',
        default='4DLung',
        choices=['all', '4DLung', 'Head', 'Pancreas'],
        help='The specific dataset name to process.'
    )

    pretrain_parser = subparsers.add_parser('pretrain', help='Pretrain LeJEPA on given dataset.')
    pretrain_parser.add_argument(
        '-t',
        '--type',
        default='full',
        choices=['full', 'sample'],
        help='The specific dataset type to pretrain on.'
    )
    pretrain_parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=1,
        help='Number of epochs to train for.'
    )
    pretrain_parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training.'
    )
    pretrain_parser.add_argument(
        '-p',
        '--patch_size',
        type=int,
        default=16,
        help='Patch size for training.'
    )
    pretrain_parser.add_argument(
        '-w',
        '--workers',
        type=int,
        default=4,
        help='Number of workers for data loading.'
    )
    pretrain_parser.add_argument(
        '-c',
        '--use_class',
        action='store_true',
        help='Whether to use CT-CBCT class or not.'
    )


    plot_parser = subparsers.add_parser('plot', help='Plot pretraining metrics.')
    plot_parser.add_argument(
        '-o',
        '--output',
        choices=['loss', 'val', 'gradient', 'time_embedding', 'all'],
        help='The metrics to plot after training.'
    )

    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess datasets')
    preprocess_parser.add_argument(
        '-n',
        '--names',
        default='4DLung',
        choices=['all', '4DLung', 'HeadNeck', 'Pancreas'],
        help='The specific dataset name to preprocess.'
    )

    downstream_parser = subparsers.add_parser('downstream', help='Downstream models.')
    downstream_parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs to train for.'
    )
    downstream_parser.add_argument(
        '-en',
        '--encode_data',
        action='store_true'
    )
    downstream_parser.add_argument(
        '-m',
        '--model',
        choices=['decoder'],
        help='Downstream model to use.'
    )
    downstream_parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=64,
        help='Size of data batches.'
    )


    args = parser.parse_args()
    config = load_config(Path('config', 'config.json'))

    if args.command == 'prep':
        Path('data').mkdir(parents=True, exist_ok=True)
        dat = CBCTDataset(
            tensors_dir=config['data'].get('data_out'),
            input_size=config['data'].get('input_size'),
            data_type=args.type,
            sample_prop=args.sample_prop,
            prop_class=args.class_prop,
            data_source=args.name,
        )
        print(f'{len(dat):,}')
        save_pickle(dat, Path('data', f'{args.type}_dataset.pkl'))
        print(f'Saved pickle file to {Path('data', f'{args.type}_dataset.pkl')}.')

    elif args.command == 'pretrain':
        pretrain(
            data_type=args.type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            num_workers=args.workers,
            use_class=args.use_class
        )
    elif args.command == 'preprocess':
        preprocess(data_dir=config['data'].get('data_dir'),
                   data_out=config['data'].get('data_out'),
                   data_name=args.names)

    elif args.command == 'plot':
        if args.output == 'loss':
            plot_loss(csv_path=Path('logs', 'losses.csv'),
                      save_dir=Path('logs'),
                      model_name='RT-JEPA',
                      save_name='training_metrics')
            
    elif args.command == 'downstream':
        if args.encode_data:
            embed_databank(
                config_path=Path('config', 'config.json'),
                jepa_weights_path=Path('model', 'rt_jepa.pt'),
                batch_size=args.batch_size,
                output_dir=Path('data')
            )

