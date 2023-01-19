from kfp import components
from kfp.components import OutputPath

def load_weights(pretrained_weights: OutputPath('Weights')):
    import gdown
    checkpoint_url = 'https://drive.google.com/drive/folders/1btey4JhgBRkoJneGKkvBLTDAuOMfPNmV?usp=share_link'
    gdown.download_folder(checkpoint_url, output=pretrained_weights, quiet=True, use_cookies=False)
    print(f'download complete!')
    print(f'checkpoint_url')

components.create_component_from_func(
    load_weights,
    output_component_file='./component-files/load_weights_component.yaml',
    packages_to_install=['gdown']
)