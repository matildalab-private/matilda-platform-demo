from kfp import components
from kfp.components import InputPath, OutputPath

def test(
    prediction_model: InputPath('TFModel'),
    trained_weights: InputPath('Weights'),
    input_img: InputPath('jpg'),
    output_img: OutputPath('jpg')
):
    import tensorflow as tf
    # import cv2
    import numpy as np

    def transform_images(x_train, size):
        x_train = tf.image.resize(x_train, (size, size))
        x_train = x_train / 255
        return x_train

    # def draw_outputs(img, outputs, class_names):
    #     boxes, objectness, classes, nums = outputs
    #     boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    #     wh = np.flip(img.shape[0:2])
    #     for i in range(nums):
    #         x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
    #         x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
    #         img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
    #         img = cv2.putText(img, '{} {:.4f}'.format(
    #             class_names[int(classes[i])], objectness[i]),
    #             x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    #     return img

    model = tf.keras.models.load_model(prediction_model)
    print('model loaded')

    model.load_weights(trained_weights+'/trained_weights.tf')
    print('trained weights loaded')

    model.summary()

    class_names = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

    img_raw = tf.image.decode_image(open(input_img, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, 416)

    print(f'img shape : {img.shape}')

    boxes, scores, classes, nums = model(img)
    print('inference done')

    def switch(tensor):
        for img_index in range(tensor.shape[0]):
            for box_index in range(tensor.shape[1]):
                xmin, ymin, xmax, ymax = tensor[img_index][box_index]
                tensor[img_index][box_index] = ymin, xmin, ymax, xmax
        return tensor

    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    boxes = switch(boxes.numpy())
    print(boxes[:,:nums[0].numpy(),:].shape, img.shape)
    img = tf.image.draw_bounding_boxes(img, boxes[:,:nums[0].numpy(),:], colors)
    from PIL import Image
    img = Image.fromarray((img.numpy()*255).astype(np.uint8)[0])
    import os
    os.mkdir(output_img)
    img.save(output_img+'/output.jpg')
    print('image written')

    # img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    # cv2.imwrite(output_img, img)

components.create_component_from_func(
    test,
    output_component_file='./component-files/pcb_test_component.yaml',
    base_image='tensorflow/tensorflow:2.2.0',
    packages_to_install=['Pillow', 'numpy']
)