import matplotlib.pyplot as plt

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [71.64, 83.37, 71.64, 71.38]

plt.figure(figsize=(8, 5))

bars = plt.bar(metrics, values)

plt.title('AgriSense AI - Phase 1 Performance Analysis')
plt.xlabel('Performance Metrics')
plt.ylabel('Score (%)')
plt.ylim(0, 100)

# Show values on top of bars
for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f'{value:.2f}%',
        ha='center'
    )

plt.tight_layout()
plt.show()