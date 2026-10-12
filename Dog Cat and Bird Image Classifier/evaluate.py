# evaluate.py
import matplotlib.pyplot as plt

def plot_history(history, title="Model Performance"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    
    # กราฟ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='orange')
    plt.title(f'{title} - Accuracy')
    plt.legend()
    
    # กราฟ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange')
    plt.title(f'{title} - Loss')
    plt.legend()
    
    plt.show()