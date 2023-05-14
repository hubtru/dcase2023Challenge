import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_encoded_data(encoded_data):
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(encoded_data)

    # Plot the reduced data
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Encoded Data Visualization')
    plt.show()