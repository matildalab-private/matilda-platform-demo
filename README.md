# matlida_platform_demo
실제 matilda platform을 사용하여 모델을 학습부터 서빙까지의 과정을 진행하기 위한 YOLOV3 기반 학습 모델입니다.
참고 : `https://github.com/zzh8829/yolov3-tf2.git`

## 실행
다음 명령어를 통하여 yolov3 모델을 학습합니다.
```
./train_run.sh [data_names]
```
현재 준비되어 있는 데이터는 PCB Defect, Brain tumor 두 종류입니다.   
data_names에는 pcb, brain 입력이 가능합니다.

|data name|reference|
|------|-----|
|brain tumor|https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets|
|pcb defect|https://www.kaggle.com/datasets/akhatova/pcb-defects|

## 결과

학습 모델은 `{data_names}_model` 디렉토리에서 확인할 수 있습니다.