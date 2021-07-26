import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam



# 케라스에서 MNIST데이터 셋을 다운로드
fashion_mnist = tf.keras.datasets.fashion_mnist

# 데이터 셋을 트레이닝 셋과 테스트 셋으로 분류
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 0부터 9까지의 라벨의 이름
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# train_images.shape # 트레인 데이터의 shape

# len(train_labels) # 트레인 데이터의 전체 길이

# 트레인 셋 6만개의 각 라벨(이름표) 0 ~ 9까지의 숫자로 라벨링
# train_labels # 총 6만개

# 테스트 데이터 셋은 10000개
# test_images.shape # 트레인 데이터와 같은 shape, 
# len(test_labels) # 트레인 데이터는 6만개지만 검증에 사용할 테스트 데이터는 1만개

plt.figure()
# 트레인 데이터셋의 첫번째 데이터 사진
plt.imshow(train_images[0])
# 이미지 옆에 컬러바 표현
plt.colorbar()
plt.grid(False)
plt.show()

# 이미지 데이터를 정규화, 0~255의 픽셀값을 0~1사이의 값으로 정규화 하는것
# 정규화를 통해 학습에 용이하도록 함
train_images = train_images / 255.0
test_images = test_images / 255.0

# 0번부터 24번까지의 25개의 이미지를 5 X 5형태로 표현
# 데이터가 제대로 준비되었는지 확인
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 모델 생성
model = Sequential()
# 분류를 위한 학습 레이어에서는 1차원 데이터로 바꾸어서 학습되어야함
# 28 X 28 형태의 2차원 shape을 1차원 데이터 형태로 변환
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation="relu"))
# softmax활성함수를 사용하여 10개의 확률로 분류, 각 확률의 합은 1
model.add(Dense(10, activation="softmax"))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(train_images, train_labels, epochs=5)

# 학습된 모델에 테스트 셋을 넣어서 loss와 정확도를 도출
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n테스트 정확도:', test_acc)

# 학습된 모델을 사용, 테스트 셋을 통해 다중분류 결과를 예측
predictions = model.predict(test_images)

# 테스트셋 전체를 모델에 넣어서 예측하였기에 1만개이 예측값이 predections에 있는것
# 1만개의 예측값 중 첫번째 테스트셋의 이미지에 대한 예측값
predictions[0]

# 첫번째 예측값중 가장 확률이 높은 라벨 도출
np.argmax(predictions[0])

# 학습된 모델을 시각화 하기 위한 함수
def plot_image(i, predictions_array, true_label, img):
    # 테스트 셋의 결과, 실제 라벨, 이미지
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    # 테스트 셋의 결과 예측 라벨
    predicted_label = np.argmax(predictions_array)
    
    # 예측 라벨이 실제 라벨과 같으면 파란색, 틀렸으면 빨간색
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    # plot의 x라벨에 예측라벨의 이름, 확률, 실제 라벨을 표현(위 if문을 통해 정답이면 파란색, 오답이면 빨간색으로 )
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)

    
# 학습된 모델을 시각화 하기 위한 함수
def plot_value_array(i, predictions_array, true_label):
    # 테스트 셋의 결과, 실제 라벨
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    # 테스트 셋의 결과 예측 0 ~ 9 항목을 softmax한 값을 회색의 bar그래프를 만듦
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    # softmax를 거치면 분류된 모델의 확률이 각 항목의 합이 1이 되도록 나오기에 0 ~ 1의 범위
    plt.ylim([0, 1])
    # 각 예측 확률 중 가장 높은(실제 모델이 예측한 항목)
    predicted_label = np.argmax(predictions_array)

    # 모델의 예측값은 빨간색, 실제 값은 파란색으로 하여 예측값이 맞으면 파란색으로 덧씌워져 파랗게 보이고
    # 예측값이 실제 값과 달랐을 경우 파란색과 빨간색 두가지가 그래프에 나타나 시각적으로 확인하기 쉽게함
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
    
# 테스트 셋을 통체로 모델에 넣었기에 몇번째 모델의 예측값을 뽑을지 정하는 변수
i = 4

plt.figure(figsize=(6,3))
# 수평으로 2개의 plot, 그중 첫번째 plot
plt.subplot(1,2,1)
# plot_image함수를 호출하여 i번째의 이미지와 예측 성공 여부, 확률을 출력
plot_image(i, predictions, test_labels, test_images)
# 수평으로 2개의 plot, 그중 두번째 plot
plt.subplot(1,2,2)
# plot_value_array함수를 호출하여 i번째의 이미지를 예측하면서 다른 항목들과의 확률 비교 시각화
plot_value_array(i, predictions,  test_labels)
plt.show()


# In[50]:





# In[51]:




