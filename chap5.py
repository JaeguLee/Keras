# %%
# Dogs vs. dogs dataset
# 훈련, 검증, 테스트 폴더로 이미지 복사하기

import os, shutil

original_dataset_dir = "./datasets/cats_and_dogs/train"

base_dir = "./datasets/cats_and_dogs_small"
train_dir = os.path.join(base_dir,"train")
validation_dir = os.path.join(base_dir,"validation")
test_dir = os.path.join(base_dir,"test")

# %%
os.mkdir(base_dir)
os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir,"cats")
train_dogs_dir = os.path.join(train_dir,"dogs")
os.mkdir(train_cats_dir)
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir,"cats")
validation_dogs_dir = os.path.join(validation_dir,"dogs")
os.mkdir(validation_cats_dir)
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir,"cats")
test_dogs_dir = os.path.join(test_dir,"dogs")
os.mkdir(test_cats_dir)
os.mkdir(test_dogs_dir)

# Cats
#.. train data
fnames:list = ["cat.{}.jpg".format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

#.. validation data
fnames:list = ["cat.{}.jpg".format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_cats_dir,fname)
    shutil.copyfile(src, dst)

#.. test data
fnames:list = ["cat.{}.jpg".format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_cats_dir,fname)
    shutil.copyfile(src, dst)

# Dogs
#.. train data
fnames:list = ["dog.{}.jpg".format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

#.. validation data
fnames:list = ["dog.{}.jpg".format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(validation_dogs_dir,fname)
    shutil.copyfile(src, dst)

#.. test data
fnames:list = ["dog.{}.jpg".format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir,fname)
    dst = os.path.join(test_dogs_dir,fname)
    shutil.copyfile(src, dst)

# %%
# test
print(f"total quantity of cats for training images : {len(os.listdir(train_cats_dir))}")
print(f"total quantity of dogs for training images : {len(os.listdir(train_dogs_dir))}")

# %%
from keras import layers
from keras import models

model = models.Sequential()
model.add(
    layers.Conv2D(32,(3,3),activation="relu",
    input_shape=(150,150,3))
)
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
# %%
model.summary()
# %%
# 모델의 훈련 설정
from keras import optimizers

model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.RMSprop(learning_rate=1e-4)
)
# %%
# pre processing
from keras.preprocessing.image import ImageDataGenerator

# 모든 이미지를 1/255로 스케일 조정
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir,  # target directory
    target_size=(150,150),  # image size 150x150
    batch_size=20,
    class_mode="binary"  #binary crossentropy 사용을 위해 이진 레이블이 필요함
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode="binary"
)
# %%
# 제너레이터 출력 예
for data_batch, labels_batch in train_generator:
    print("data 크기",data_batch.shape)
    print("label 크기",labels_batch.shape)
    break
# %%
# 배치 제너레이터를 사용하여 모델 훈련하기
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)
# %%
model.save('cats_and_dogs_small_1.h5')
# %%
import matplotlib.pyplot as plt

#acc = history.history["acc"]
#val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(loss) + 1)
"""
plt.plot(epochs,acc,'bo',label="Training acc")
plt.plot(epochs,val_acc,'bo',label="Training validation acc")
plt.title("Trainign and validation accuracy")
plt.legend()
"""
plt.figure()

plt.plot(epochs,loss,'bo',label="Training loss")
plt.plot(epochs,val_loss,'b',label="Training validation loss")
plt.title("Trainign and validation loss")
plt.legend()

plt.show()

# 5.2.5부터 시작
# %%
history.history.keys()
# %%
