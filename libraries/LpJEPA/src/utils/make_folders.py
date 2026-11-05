import json
from pathlib import Path


def make_folders():
    config_path = Path('../../../config')
    config_path.mkdir(parents=True, exist_ok=True)
    Path('../../../logs').mkdir(parents=True, exist_ok=True)
    Path('../rt_jepa/model/outputs').mkdir(parents=True, exist_ok=True)
    Path('../rt_jepa/model/weights').mkdir(parents=True, exist_ok=True)

    if not config_path.iterdir():
        config_dict = {
            'data': {
                'data_dir': 'Volumes/diskAshurM2/CBCT_enhancement/manifest-ObLxS9Wd1073675925233948759/4D-Lung',
                'input_size': [512, 512],
            },
            'model': {'model_dir': 'model'},
        }
        Path(config_path / 'config.json').write_text(json.dumps(config_dict, indent=2))
