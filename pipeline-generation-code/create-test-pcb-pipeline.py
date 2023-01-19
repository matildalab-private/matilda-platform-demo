import kfp
from kfp import components, dsl

load_train_data_op = components.load_component_from_file('./component-files/load_pcb_train_data_component.yaml')

load_weights_op = components.load_component_from_file('./component-files/load_weights_component.yaml')

create_model_op = components.load_component_from_file('./component-files/create_model_component.yaml')

update_model_weights_op = components.load_component_from_file('./component-files/update_model_weights_component.yaml')

train_model_op = components.load_component_from_file('./component-files/train_pcb_model_component.yaml')

load_test_img_op = components.load_component_from_file('./component-files/load_pcb_test_img_component.yaml')

test_op = components.load_component_from_file('./component-files/pcb_test_component.yaml')

@dsl.pipeline(name='test YOLOv3 pipeline')
def test_pipeline():
    load_data_task = load_train_data_op()
    load_weights_task = load_weights_op()
    create_model_task = create_model_op()

    update_model_task = update_model_weights_op(
        load_weights_task.outputs['pretrained_weights'],
        create_model_task.outputs['compiled_model']
    )

    train_model_task = train_model_op(
        load_data_task.outputs['train_dataset'],
        update_model_task.outputs['loaded_model']
    )

    load_test_img_task = load_test_img_op()

    test_task = test_op(
        create_model_task.outputs['prediction_model'],
        train_model_task.outputs['trained_weights'],
        load_test_img_task.outputs['input_img']
    )

kfp.compiler.Compiler().compile(test_pipeline, 'test_pipeline.yaml')