# MNIST
다중이미지 분류 예제

### 구조
 - Tensorflow Keras의 MNIST데이터 셋을 사용
 - 28 X 28 크기의 흑백 fashion 사진으로 Train셋 6만장을 통해 학습
 - 각 이미지는 총 10개의 카테고리로 구성
 - 학습이 완료된 모델은 입력받은 이미지를 확률에 따라 카테고리 분류

### 학습
 - 데이터는 0 ~ 9까지의 숫자로 라벨링 되어있기에 자연어로 알아보기 쉽게 카테고리 리스트 제작
 - 28 X 28의 2차원 구조고 흑백이미지이기에 255.0으로 나눠서 0 ~ 255 픽셀값을 0 ~ 1의 값으로 정규화 작업 진행
 - 학습 모델을 생성하는데 있어 이미지학습에서는 2차원 구조의 데이터를 1차원 구조로 변환시켜줘야 하므로 Flatten을 사용하여 인풋데이터를 1차원 구조로 변환
 - 로스함수는 다중분류에 사용하는 sparse_categorical_crossentropy를 사용
 - 학습이 완료된 모델을 Matplotlib을 사용하여 시각화해 모델의 테스트를 진행

### 결과
 - 학습모델의 마지막 아웃풋레이어의 활성화함수를 softmax로 하여 각 카테고리의 확률을 토대로 다중분류를 하였음
 - 각 카테고리에 해당될 확률을 내포하고있었고, 각 확률의 총 합은 1
 - 이미지의 크기가 작고 많은 데이터 양으로 짧은 시간에 정확한 인공지능모델을 테스트해볼 수 있었음
<img src="https://user-images.githubusercontent.com/87750521/127043668-43396974-b14a-44e6-ac7a-9645cd1b4d35.png" width="360" height="200">
