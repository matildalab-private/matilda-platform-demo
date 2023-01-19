from kfp import components
from kfp.components import OutputPath

def load_train_data(train_dataset: OutputPath('Dataset')):
    import gdown
    dataset_url = 'https://drive.google.com/file/d/1qn0mLFV7NBbmw6-ZTuF_tN_oJxRHx-XR/view?usp=share_link'
    gdown.download(dataset_url, output=train_dataset, quiet=True, fuzzy=True)
    print(f'download complete!')

components.create_component_from_func(
    load_train_data, 
    output_component_file='./component-files/load_pcb_train_data_component.yaml',
    packages_to_install=['gdown']
)