# config.py

# ขนาดรูปภาพที่จะนำเข้าโมเดล
IMG_SIZE = (224, 224)

# จำนวนรูปภาพที่ประมวลผลในแต่ละรอบ
BATCH_SIZE = 32

# จำนวนรอบในการเทรนโมเดล
EPOCHS = 15

# จำนวนคลาสของสัตว์ (Dog, Cat, Bird)
NUM_CLASSES = 3

# พาธของโฟลเดอร์ข้อมูล
TRAIN_DIR = 'train_images'
VAL_DIR = 'test_images'