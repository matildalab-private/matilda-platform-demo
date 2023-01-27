from kfp.components import InputPath, OutputPath
from typing import List
import kfp
from kfp import components, dsl
import json


def load_train_data(
    train_dataset_url: str, 
    val_dataset_url: str, 
    train_dataset: OutputPath('Dataset'), 
    val_dataset: OutputPath('Dataset')
):
    import gdown
    gdown.download(train_dataset_url, output=train_dataset, quiet=True, fuzzy=True)
    gdown.download(val_dataset_url, output=val_dataset, quiet=True, fuzzy=True)
    print(f'download complete!')

load_train_data_op = components.create_component_from_func(
    load_train_data, 
    # output_component_file='./component-files-yaml/load_train_data_component.yaml',
    packages_to_install=['gdown']
)


def load_test_img(img_url: str, input_img: OutputPath('jpg')):
    import gdown
    gdown.download(img_url, output=input_img, quiet=True, fuzzy=True)
    print(f'download complete!')

load_test_img_op = components.create_component_from_func(
    load_test_img, 
    # output_component_file='./component-files-yaml/load_test_img_component.yaml',
    packages_to_install=['gdown']
)


def load_weights(checkpoint_url: str, pretrained_weights: OutputPath('Weights')):
    import gdown
    gdown.download_folder(checkpoint_url, output=pretrained_weights, quiet=True, use_cookies=False)
    print(f'download complete!')

load_weights_op = components.create_component_from_func(
    load_weights,
    # output_component_file='./component-files-yaml/load_weights_component.yaml',
    packages_to_install=['gdown']
)


def train_model(
    model_size: int,
    num_classes: int,
    num_epohcs: int,
    class_names: List[str],
    checkpoint_name: str,
    pretrained_weights: InputPath('Weights'),
    train_dataset: InputPath('Dataset'),
    val_dataset: InputPath('Dataset'),
    trained_weights: OutputPath('Weights')
):
    from yolov3_minimal import load_tfrecord_dataset, transform_images, transform_targets, YoloV3, YoloLoss, freeze_all
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        ReduceLROnPlateau,
        EarlyStopping,
        ModelCheckpoint
    )

    SIZE = model_size
    NUM_CLASSES = num_classes
    NUM_EPOCHS = num_epohcs
    LEARNING_RATE = 1e-3

    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),(59, 119), (116, 90), (156, 198), (373, 326)],np.float32) / 416
    anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    train_dataset = load_tfrecord_dataset(train_dataset, class_names, SIZE)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(8)
    train_dataset = train_dataset.map(lambda x, y: (
        transform_images(x, SIZE),
        transform_targets(y, anchors, anchor_masks, SIZE)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = load_tfrecord_dataset(val_dataset, class_names, SIZE)
    val_dataset = val_dataset.batch(8)
    val_dataset = val_dataset.map(lambda x, y: (
        transform_images(x, SIZE),
        transform_targets(y, anchors, anchor_masks, SIZE)))
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    model = YoloV3(SIZE, classes=NUM_CLASSES, training=True)
    model.load_weights(pretrained_weights+'/'+checkpoint_name).expect_partial()
    freeze_all(model.get_layer('yolo_darknet'))

    optimizer = tf.keras.optimizers.legacy.Adam(lr=LEARNING_RATE)
    loss = [YoloLoss(anchors[mask], classes=NUM_CLASSES) for mask in anchor_masks]
    model.compile(optimizer=optimizer, loss=loss)

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=5, verbose=1),
        ModelCheckpoint(
            filepath=trained_weights+'/trained_weights.tf',
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        )
    ]

    import os
    os.mkdir(trained_weights)

    model.fit(train_dataset, epochs=NUM_EPOCHS, callbacks=callbacks, validation_data=val_dataset)    

train_model_op = components.create_component_from_func(
    train_model,
    # output_component_file='./component-files-yaml/train_model_component.yaml',
    packages_to_install=['yolov3-minimal']
)


def test(
    model_size: int,
    num_classes: int,
    class_names: List[str],
    trained_weights: InputPath('Weights'),
    input_img: InputPath('jpg'),
    output_img: OutputPath('jpg')
):
    from yolov3_minimal import transform_images, draw_outputs, YoloV3
    import tensorflow as tf
    import cv2
    import os

    SIZE = model_size
    NUM_CLASSES = num_classes

    model = YoloV3(SIZE, classes=NUM_CLASSES)
    model.load_weights(trained_weights+'/trained_weights.tf').expect_partial()
    print('trained weights loaded')

    img_raw = tf.image.decode_image(open(input_img, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, SIZE)

    boxes, scores, classes, nums = model(img)
    print('inference done')

    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    os.mkdir(output_img)
    cv2.imwrite(output_img+'/output.jpg', img)

test_op = components.create_component_from_func(
    test,
    # output_component_file='./component-files-yaml/test_component.yaml',
    packages_to_install=['yolov3-minimal']
)

def serve(temp_var: InputPath('jpg')):
    if temp_var:
        print('Model served successfully.')
    else:
        print('There was an error serving the model.')

serve_op = components.create_component_from_func(
    serve,
    # output_component_file='./component-files-yaml/serve_component.yaml'
)

@dsl.pipeline(name='YOLOv3 pipeline')
def yolov3_pipeline(
    # brain tumor
    # train_dataset_url="https://drive.google.com/file/d/1Sq0bph5QJE5U_x-qu8hUcjgiTONeBDy1/view?usp=sharing",
    # val_dataset_url="https://drive.google.com/file/d/172vMkaGKkol2x1juNzjWdEwNZrTyZnvz/view?usp=share_link",
    # checkpoint_url="https://drive.google.com/drive/folders/1-C3N6h-CtdojHjEyFvXGXBorDFBo_k4z?usp=share_link",
    # checkpoint_name="axial_ckpt.tf",
    # test_img_url="https://drive.google.com/file/d/13PzBr8jBjHdt4VMaEcEOkMoCF6VKpmx1/view?usp=share_link",
    # model_size='256',
    # num_classes='2',
    # num_epochs='1',
    # class_names='["negative", "positive"]'
    # pcb defect
    train_dataset_url="https://drive.google.com/file/d/1qn0mLFV7NBbmw6-ZTuF_tN_oJxRHx-XR/view?usp=sharing",
    val_dataset_url="https://drive.google.com/file/d/1gUCWhls3ZyVurdYFl1iikRUlDHmKi4ZQ/view?usp=sharing",
    checkpoint_url="https://drive.google.com/drive/folders/1btey4JhgBRkoJneGKkvBLTDAuOMfPNmV?usp=share_link",
    checkpoint_name="yolov3_train_30.tf",
    test_img_url="https://drive.google.com/file/d/13SNqfX3z8N1-qt4PwaS4RL0O2SydN09j/view?usp=sharing",
    model_size='416',
    num_classes='6',
    num_epochs='1',
    class_names='[‘missing_hole’, ‘mouse_bite’, ‘open_circuit’, ‘short’, ‘spur’, ‘spurious_copper’]'
):

    load_data_task = load_train_data_op(train_dataset_url, val_dataset_url)
    load_weights_task = load_weights_op(checkpoint_url)

    train_model_task = train_model_op(
        model_size,
        num_classes,
        num_epochs,
        class_names,
        checkpoint_name,
        load_weights_task.outputs['pretrained_weights'],
        load_data_task.outputs['train_dataset'],
        load_data_task.outputs['val_dataset'],
    )

    load_test_img_task = load_test_img_op(test_img_url)

    test_task = test_op(
        model_size,
        num_classes,
        class_names,
        train_model_task.outputs['trained_weights'],
        load_test_img_task.outputs['input_img']
    )

    serve_op(test_task.output)

kfp.compiler.Compiler().compile(yolov3_pipeline, 'pipeline.yaml')