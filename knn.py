from math import sqrt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def getting_neighbors(k, weight, height, data_list):
    distances = []
    for value in data_list:
        df_height = value[3]
        df_weights = value[4]
        df_sizes = value[1]
        distance = sqrt((weight - df_height) ** 2 + (height - df_weights) ** 2)
        distances.append({'distance': distance, 'size': df_sizes})
    distances.sort(key=lambda d: d['distance'])
    return distances[:k]


def get_prediction(neighbors):
    sizes = {}
    for distance_item in neighbors:
        size = distance_item['size']
        if size in sizes:
            sizes[size] += 1
        else:
            sizes[size] = 1
    k = len(neighbors)
    return {size: value / k * 100 for size, value in sizes.items()}


def main():
    male = pd.read_csv('male_completed')
    female = pd.read_csv('female_completed')

    data_list = male.values.tolist() + female.values.tolist()

    n = int(sqrt(len(data_list)))
    k_values = [1, 3, 5, 7, n]

    # gender = input("Male or Female?: ")
    height = int(input("Whats your height private!?: "))
    weight = int(input("How much do you weigh private!?: "))

    for k in k_values:
        neighbors = getting_neighbors(k, weight, height, data_list)
        result = get_prediction(neighbors)
        print(f'I estimate that you should get a tshirt size of (k={k})')
        for size, proc in result.items():
            print(f'\t{size} Im a confident of {proc:.2f}%')
        print('================================================')

        heights = [data[3] for data in data_list]
        weights = [data[4] for data in data_list]
        labels = [data[1] for data in data_list]

        data = list(zip(weights, heights))
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(data, labels)
        print(f'lets see what the machine predicts:{knn_model.predict([[weight, height]])}for k={k}')
        print('================================================')

    x = male['weightkg']
    y = male['stature']

    neighbors = getting_neighbors(weight=weight, height=height, data_list=data_list, k=k)
    predict = get_prediction(neighbors)
    plt.scatter(x, y)
    plt.scatter(weight, height, edgecolors='black')
    plt.show()


if __name__ == '__main__':
    main()
