import csv
import matplotlib.pyplot as plt
def plot_data(csv_file_path: str) -> None:
    precision = []
    recall = []
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            precision.append(float(row[0]))
            recall.append(float(row[1]))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()
# Usage example:
plot_data('data_file.csv')
