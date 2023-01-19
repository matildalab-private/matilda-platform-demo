if [[ "$#" -gt 1 ||  "$#" -lt 1 ]]; then
  echo "Usage: $0 DATASET_NAME"
  exit
fi

DATASET_NAME=$1
echo "run generation ${DATASET_NAME} detection model"

echo 'env settings ... '
pip install -r requirements.txt

if [ $? -eq 0 ];then

  if [[ "$1" =~ pcb ]]; then

    echo 'download yolov3 weights ... '
    mkdir data
    wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
    python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf

    echo 'download pcb dataset'
    gdown https://drive.google.com/drive/folders/1ES9wMg9S0lmuy084-zFWc3Tbd8RTwIWx?usp=share_link --folder -O ./pcb_dataset

    echo 'start pcb defect detection model train ... '
    python train.py \
        --names pcb \
        --batch_size 8 \
        --dataset ./pcb_dataset/pcb_dataset.tfrecord \
        --epochs 2 --classes ./pcb_dataset/pcb.names \
        --num_classes 6 --size 416 \
        --mode eager_tf \
        --transfer darknet \
        --weights ./checkpoints/yolov3.tf \
        --output ./pcb_defect_model \
        --weights_num_classes 80

  elif [[ "$1" =~ brain ]]; then

    echo 'download yolov3 weights ... '
    mkdir data
    wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
    python convert.py --weights ./data/yolov3.weights --output ./checkpoints/yolov3.tf

    echo 'download pretrained weights and dataset ... '
    gdown https://drive.google.com/drive/folders/1DTLqPKm-eypYlyD54s-sEkcL-yGCPsyo?usp=share_link --folder -O ./brain_dataset

    echo 'start brain tumor detection model train ... '
    python train.py \
        --names brain \
        --batch_size 8 \
        --dataset ./brain_dataset/axial_brain_train.tfrecord  \
        --val_dataset ./brain_dataset/axial_brain_val.tfrecord
        --epochs 2 --size 256 \
        --classes ./brain_dataset/class.txt \
        --num_classes 2 \
        --mode eager_tf \
        --transfer darknet \
        --weights ./checkpoints/yolov3.tf \
        --output ./brain_tumor_model \
        --weights_num_classes 80

  else
    echo "please input dataset name [pcb, brain]"
    exit 9
  fi

else
  echo "install failed"
  exit 9
fi


echo "finished ..."