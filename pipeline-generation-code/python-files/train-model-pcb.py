from kfp import components
from kfp.components import InputPath, OutputPath

def train_model(
    train_dataset: InputPath('Dataset'),
    loaded_model: InputPath('TFModel'),
    trained_weights: OutputPath('Weights')
):
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        ReduceLROnPlateau,
        EarlyStopping,
        ModelCheckpoint
    )
    from tensorflow.keras.losses import (
        binary_crossentropy,
        sparse_categorical_crossentropy
    )

    def _meshgrid(n_a, n_b):
        return [
            tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
            tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
        ]

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

    def YoloLoss(anchors, classes=6, ignore_thresh=0.5):
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

    IMAGE_FEATURE_MAP = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
    }

    def parse_tfrecord(tfrecord, class_table, size):
        x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
        x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        x_train = tf.image.resize(x_train, (size, size))

        class_text = tf.sparse.to_dense(
            x['image/object/class/text'], default_value='')
        labels = tf.cast(class_table.lookup(class_text), tf.float32)
        y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                            tf.sparse.to_dense(x['image/object/bbox/ymin']),
                            tf.sparse.to_dense(x['image/object/bbox/xmax']),
                            tf.sparse.to_dense(x['image/object/bbox/ymax']),
                            labels], axis=1)

        paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]
        y_train = tf.pad(y_train, paddings)

        return x_train, y_train

    def load_tfrecord_dataset(file_pattern, size=416):
        keys_tensor = tf.constant(['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper'])
        vals_tensor = tf.constant([0, 1, 2, 3, 4, 5])
        init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
        class_table = tf.lookup.StaticHashTable(init, default_value=-1)
        # LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
        # class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        #     class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)
        files = tf.data.Dataset.list_files(file_pattern)
        dataset = files.flat_map(tf.data.TFRecordDataset)
        return dataset.map(lambda x: parse_tfrecord(x, class_table, size))

    @tf.function
    def transform_targets_for_output(y_true, grid_size, anchor_idxs):
        # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
        N = tf.shape(y_true)[0]

        # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
        y_true_out = tf.zeros(
            (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

        anchor_idxs = tf.cast(anchor_idxs, tf.int32)

        indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
        updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
        idx = 0
        for i in tf.range(N):
            for j in tf.range(tf.shape(y_true)[1]):
                if tf.equal(y_true[i][j][2], 0):
                    continue
                anchor_eq = tf.equal(
                    anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

                if tf.reduce_any(anchor_eq):
                    box = y_true[i][j][0:4]
                    box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                    anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                    grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                    # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                    indexes = indexes.write(
                        idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                    updates = updates.write(
                        idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                    idx += 1

        # tf.print(indexes.stack())
        # tf.print(updates.stack())

        return tf.tensor_scatter_nd_update(
            y_true_out, indexes.stack(), updates.stack())
    
    def transform_targets(y_train, anchors, anchor_masks, size):
        y_outs = []
        grid_size = size // 32

        # calculate anchor index for true boxes
        anchors = tf.cast(anchors, tf.float32)
        anchor_area = anchors[..., 0] * anchors[..., 1]
        box_wh = y_train[..., 2:4] - y_train[..., 0:2]
        box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                        (1, 1, tf.shape(anchors)[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
            tf.minimum(box_wh[..., 1], anchors[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
        anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

        y_train = tf.concat([y_train, anchor_idx], axis=-1)

        for anchor_idxs in anchor_masks:
            y_outs.append(transform_targets_for_output(
                y_train, grid_size, anchor_idxs))
            grid_size *= 2

        return tuple(y_outs)


    def transform_images(x_train, size):
        x_train = tf.image.resize(x_train, (size, size))
        x_train = x_train / 255
        return x_train
    
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),(59, 119), (116, 90), (156, 198), (373, 326)],np.float32) / 416
    anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

    train_dataset = load_tfrecord_dataset(train_dataset)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(8)
    train_dataset = train_dataset.map(lambda x, y: (
        transform_images(x, 416),
        transform_targets(y, anchors, anchor_masks, 416)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    model = tf.keras.models.load_model(loaded_model, custom_objects={'yolo_loss':[[YoloLoss(anchors[mask], classes=6) for mask in anchor_masks]]})

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=5, verbose=1),
    ]

    import os
    os.mkdir(trained_weights)

    model.fit(train_dataset, epochs=1, callbacks=callbacks)    
    model.save_weights(trained_weights+'/trained_weights.tf')
    print(os.listdir('tmp/outputs/trained_weights/data/'))

components.create_component_from_func(
    train_model,
    output_component_file='./component-files/train_pcb_model_component.yaml',
    base_image='tensorflow/tensorflow:2.2.0',
    packages_to_install=['numpy']
)
