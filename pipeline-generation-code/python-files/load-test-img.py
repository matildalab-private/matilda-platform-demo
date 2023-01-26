from kfp import components
from kfp.components import OutputPath

def load_test_img(input_img: OutputPath('jpg')):
    import gdown
    img_url = 'https://drive.google.com/file/d/13SNqfX3z8N1-qt4PwaS4RL0O2SydN09j/view?usp=share_link'
    gdown.download(img_url, output=input_img, quiet=True, fuzzy=True)
    print(f'download complete!')
    

components.create_component_from_func(
    load_test_img, 
    output_component_file='./component-files/load_test_img_component.yaml',
    packages_to_install=['gdown']
)