# train.py
from dataset import get_data_generators
from models import build_custom_cnn, build_transfer_model
from evaluate import plot_history
import config

def main():
    # 1. โหลดข้อมูล
    print("Loading data...")
    train_gen, val_gen = get_data_generators()

    # 2. เทรนโมเดลแบบสร้างเอง (CNN)
    print("\n=== Training Custom CNN ===")
    cnn_model = build_custom_cnn()
    history_cnn = cnn_model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen
    )
    cnn_model.save('custom_cnn_model.h5')
    plot_history(history_cnn, title="Custom CNN")

    # 3. เทรนโมเดล Transfer Learning (MobileNetV2)
    print("\n=== Training Transfer Learning (MobileNetV2) ===")
    tl_model = build_transfer_model()
    history_tl = tl_model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen
    )
    tl_model.save('transfer_model.h5')
    plot_history(history_tl, title="Transfer Learning")
    
    print("\nTraining completed! Models saved as .h5 files.")

if __name__ == '__main__':
    main()