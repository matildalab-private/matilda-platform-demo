from kfp import components
from kfp.components import OutputPath

def create_model(compiled_model: OutputPath('TFModel'), prediction_model: OutputPath('TFModel')):
    import json
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (
        Layer,
        Add,
        Concatenate,
        Conv2D,
        Input,
        Lambda,
        LeakyReLU,
        MaxPool2D,
        UpSampling2D,
        ZeroPadding2D,
        BatchNormalization,
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.losses import (
        binary_crossentropy,
        sparse_categorical_crossentropy
    )

    SIZE = 416
    NUM_CLASSES = 6
    LEARNING_RATE = 1e-3

    def broadcast_iou(box_1, box_2):
        # box_1: (..., (x1, y1, x2, y2))
        # box_2: (N, (x1, y1, x2, y2))

        # broadcast boxes
        box_1 = tf.expand_dims(box_1, -2)
        box_2 = tf.expand_dims(box_2, 0)
        # new_shape: (..., N, (x1, y1, x2, y2))
        new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
        box_1 = tf.broadcast_to(box_1, new_shape)
        box_2 = tf.broadcast_to(box_2, new_shape)

        int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                        tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
        int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                        tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
        int_area = int_w * int_h
        box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
            (box_1[..., 3] - box_1[..., 1])
        box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
            (box_2[..., 3] - box_2[..., 1])
        return int_area / (box_1_area + box_2_area - int_area)

    yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),(59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
    yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    def DarknetConv(x, filters, size, strides=1, batch_norm=True):
        if strides == 1:
            padding = 'same'
        else:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
            padding = 'valid'
        x = Conv2D(filters=filters, kernel_size=size,
                strides=strides, padding=padding,
                use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
        if batch_norm:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x


    def DarknetResidual(x, filters):
        prev = x
        x = DarknetConv(x, filters // 2, 1)
        x = DarknetConv(x, filters, 3)
        x = Add()([prev, x])
        return x

    def DarknetBlock(x, filters, blocks):
        x = DarknetConv(x, filters, 3, strides=2)
        for _ in range(blocks):
            x = DarknetResidual(x, filters)
        return x

    def Darknet(name=None):
        x = inputs = Input([None, None, 3])
        x = DarknetConv(x, 32, 3)
        x = DarknetBlock(x, 64, 1)
        x = DarknetBlock(x, 128, 2)  # skip connection
        x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
        x = x_61 = DarknetBlock(x, 512, 8)
        x = DarknetBlock(x, 1024, 4)
        return Model(inputs, (x_36, x_61, x), name=name)

    def YoloConv(filters, name=None):
        def yolo_conv(x_in):
            if isinstance(x_in, tuple):
                inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
                x, x_skip = inputs

                # concat with skip connection
                x = DarknetConv(x, filters, 1)
                x = UpSampling2D(2)(x)
                x = Concatenate()([x, x_skip])
            else:
                x = inputs = Input(x_in.shape[1:])

            x = DarknetConv(x, filters, 1)
            x = DarknetConv(x, filters * 2, 3)
            x = DarknetConv(x, filters, 1)
            x = DarknetConv(x, filters * 2, 3)
            x = DarknetConv(x, filters, 1)
            return Model(inputs, x, name=name)(x_in)
        return yolo_conv
    
    class ReshapeLayer(Layer):
        def __init__(self, anchors, classes):
            super(ReshapeLayer, self).__init__()
            self.anchors = anchors
            self.classes = classes
        
        def call(self, inputs):
            return tf.reshape(inputs, (-1, tf.shape(inputs)[1], tf.shape(inputs)[2], self.anchors, self.classes + 5))

    def YoloOutput(filters, anchors, classes, name=None):
        def yolo_output(x_in):
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters * 2, 3)
            x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
            # x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
            x = ReshapeLayer(anchors, classes)(x)
            return Model(inputs, x, name=name)(x_in)
        return yolo_output

    def _meshgrid(n_a, n_b):
        return [
            tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
            tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
        ]

    def yolo_boxes(pred, anchors, classes):
        # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        grid_size = tf.shape(pred)[1:3]
        box_xy, box_wh, objectness, class_probs = tf.split(
            pred, (2, 2, 1, classes), axis=-1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        grid = _meshgrid(grid_size[1],grid_size[0])
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
            tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return bbox, objectness, class_probs, pred_box

    
    def yolo_nms(outputs, anchors, masks, classes):
        # boxes, conf, type
        b, c, t = [], [], []

        for o in outputs:
            b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
            c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
            t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

        bbox = tf.concat(b, axis=1)
        confidence = tf.concat(c, axis=1)
        class_probs = tf.concat(t, axis=1)

        # If we only have one class, do not multiply by class_prob (always 0.5)
        if classes == 1:
            scores = confidence
        else:
            scores = confidence * class_probs

        dscores = tf.squeeze(scores, axis=0)
        scores = tf.reduce_max(dscores,[1])
        bbox = tf.reshape(bbox,(-1,4))
        classes = tf.argmax(dscores,1)
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
            boxes=bbox,
            scores=scores,
            max_output_size=100,
            iou_threshold=0.5,
            score_threshold=0.5,
            soft_nms_sigma=0.5
        )
        
        num_valid_nms_boxes = tf.shape(selected_indices)[0]

        selected_indices = tf.concat([selected_indices,tf.zeros(100-num_valid_nms_boxes, tf.int32)], 0)
        selected_scores = tf.concat([selected_scores,tf.zeros(100-num_valid_nms_boxes,tf.float32)], -1)

        boxes=tf.gather(bbox, selected_indices)
        boxes = tf.expand_dims(boxes, axis=0)
        scores=selected_scores
        scores = tf.expand_dims(scores, axis=0)
        classes = tf.gather(classes,selected_indices)
        classes = tf.expand_dims(classes, axis=0)
        valid_detections=num_valid_nms_boxes
        valid_detections = tf.expand_dims(valid_detections, axis=0)

        return boxes, scores, classes, valid_detections

    class YoloBoxLayer(Layer):
        def __init__(self, anchors, classes=NUM_CLASSES, name=None):
            super(YoloBoxLayer, self).__init__(name=name)
            self.anchors = anchors
            self.classes = classes

        def __call__(self, inputs):
            return yolo_boxes(inputs, self.anchors, self.classes)
    
    class YoloNMS(Layer):
        def __init__(self, anchors, masks, classes, name=None):
            super(YoloNMS, self).__init__(name=name)
            self.anchors = anchors
            self.masks = masks
            self.classes = classes
        
        def __call__(self, inputs):
            return yolo_nms(inputs, self.anchors, self.masks, self.classes)


    def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=NUM_CLASSES, training=False):

        x = inputs = Input([size, size, channels], name='input')

        x_36, x_61, x = Darknet(name='yolo_darknet')(x)

        x = YoloConv(512, name='yolo_conv_0')(x)
        output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

        x = YoloConv(256, name='yolo_conv_1')((x, x_61))
        output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

        x = YoloConv(128, name='yolo_conv_2')((x, x_36))
        output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

        if training:
            return Model(inputs, (output_0, output_1, output_2), name='yolov3')
    
        anchors_0 = np.array([(116, 90), (156, 198), (373, 326)], np.float32) / 416
        anchors_1 = np.array([(30, 61), (62, 45),(59, 119)], np.float32) / 416
        anchors_2 = np.array([(10, 13), (16, 30), (33, 23)], np.float32) / 416

        boxes_0 = YoloBoxLayer(anchors_0, name='yolo_boxes_0')(output_0)
        boxes_1 = YoloBoxLayer(anchors_1, name='yolo_boxes_1')(output_1)
        boxes_2 = YoloBoxLayer(anchors_2, name='yolo_boxes_2')(output_2)

        outputs = YoloNMS(anchors, masks, classes, name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

        return Model(inputs, outputs, name='yolov3')
    
    def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
        def yolo_loss(y_true, y_pred):
            # 1. transform all pred outputs
            # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
            pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
                y_pred, anchors, classes)
            pred_xy = pred_xywh[..., 0:2]
            pred_wh = pred_xywh[..., 2:4]

            # 2. transform all true outputs
            # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
            true_box, true_obj, true_class_idx = tf.split(
                y_true, (4, 1, 1), axis=-1)
            true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
            true_wh = true_box[..., 2:4] - true_box[..., 0:2]

            # give higher weights to small boxes
            box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

            # 3. inverting the pred box equations
            grid_size = tf.shape(y_true)[1]
            grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
            grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
            true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
                tf.cast(grid, tf.float32)
            true_wh = tf.math.log(true_wh / anchors)
            true_wh = tf.where(tf.math.is_inf(true_wh),
                            tf.zeros_like(true_wh), true_wh)

            # 4. calculate all masks
            obj_mask = tf.squeeze(true_obj, -1)
            # ignore false positive when iou is over threshold
            best_iou = tf.map_fn(
                lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                    x[1], tf.cast(x[2], tf.bool))), axis=-1),
                (pred_box, true_box, obj_mask),
                tf.float32)
            ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

            # 5. calculate all losses
            xy_loss = obj_mask * box_loss_scale * \
                tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
            wh_loss = obj_mask * box_loss_scale * \
                tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
            obj_loss = binary_crossentropy(true_obj, pred_obj)
            obj_loss = obj_mask * obj_loss + \
                (1 - obj_mask) * ignore_mask * obj_loss
            # TODO: use binary_crossentropy instead
            class_loss = obj_mask * sparse_categorical_crossentropy(
                true_class_idx, pred_class)

            # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
            xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
            wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
            obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
            class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

            return xy_loss + wh_loss + obj_loss + class_loss
        return yolo_loss

    model_t = YoloV3(SIZE, channels=3, training=True, classes=NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = [YoloLoss(yolo_anchors[mask], classes=NUM_CLASSES) for mask in yolo_anchor_masks]
    model_t.compile(optimizer=optimizer, loss=loss)
    model_t.save(compiled_model)

    model_p = YoloV3(SIZE, classes=NUM_CLASSES)
    model_p.save(prediction_model)

    model_p.summary()
    model_t.summary()

components.create_component_from_func(
    create_model,
    output_component_file='./component-files/create_model_component.yaml',
    base_image='tensorflow/tensorflow:2.2.0',
    packages_to_install=['numpy']
)