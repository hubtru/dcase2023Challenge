import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_encoded_data(train_data, test_data):
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_train_data = pca.fit_transform(train_data)
    reduced_test_data = pca.fit_transform(test_data)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the reduced encoded data
    axs[0].scatter(reduced_train_data[:, 0], reduced_train_data[:, 1])
    axs[0].set_xlabel('Principal Component 1')
    axs[0].set_ylabel('Principal Component 2')
    axs[0].set_title('Train Data Visualization')

    # Plot the reduced new data
    axs[1].scatter(reduced_test_data[:, 0], reduced_test_data[:, 1])
    axs[1].set_xlabel('Principal Component 1')
    axs[1].set_ylabel('Principal Component 2')
    axs[1].set_title('Test_data Visualization')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

