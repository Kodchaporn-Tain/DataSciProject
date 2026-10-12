# dataset.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

def get_data_generators():
    # เพิ่มความหลากหลายให้ภาพชุด Train (หมุน, ขยับ, พลิก)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # ชุด Validation ทำแค่ปรับสเกลค่าสีเป็น 0-1
    val_datagen = ImageDataGenerator(rescale=1./255)

    # โหลดภาพจากโฟลเดอร์ train_images
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )

    # โหลดภาพจากโฟลเดอร์ test_images
    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )
    
    return train_generator, val_generator