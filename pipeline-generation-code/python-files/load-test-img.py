from kfp import components
from kfp.components import OutputPath

def load_test_img(input_img: OutputPath('jpg')):
    import gdown
    img_url = 'https://drive.google.com/file/d/13PzBr8jBjHdt4VMaEcEOkMoCF6VKpmx1/view?usp=share_link'
    gdown.download(img_url, output=input_img, quiet=True, fuzzy=True)
    print(f'download complete!')

components.create_component_from_func(
    load_test_img, 
    output_component_file='./component-files/load_test_img_component.yaml',
    packages_to_install=['gdown']
)