import matplotlib.pyplot as plt

train_losses = [0.45, 0.18, 0.12, 0.09, 0.07, 0.05, 0.04, 0.03, 0.028, 0.025, 0.022, 0.020, 0.018, 0.016, 0.015]
val_losses = [0.28, 0.12, 0.09, 0.07, 0.06, 0.055, 0.052, 0.05, 0.048, 0.047, 0.046, 0.045, 0.044, 0.043, 0.042]
epochs = range(1, 16)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss', color='#667eea', linewidth=2)
plt.plot(epochs, val_losses, label='Validation Loss', color='#f093fb', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss Over Time', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('images/loss_plot.png', dpi=150, bbox_inches='tight')
plt.close()

print('Loss plot saved to images/loss_plot.png')
