import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Simulate image-like feature data
np.random.seed(42)

# Generate synthetic data for "cats" and "dogs"
num_samples = 100

# "Cats" feature vectors (e.g., lower values for simplicity)
cat_features = np.random.normal(loc=20, scale=5, size=(num_samples, 2))  # 2 features
cat_labels = np.zeros(num_samples)  # Label: 0 for cats

# "Dogs" feature vectors (e.g., higher values for simplicity)
dog_features = np.random.normal(loc=40, scale=5, size=(num_samples, 2))  # 2 features
dog_labels = np.ones(num_samples)  # Label: 1 for dogs

# Combine features and labels
features = np.vstack((cat_features, dog_features))
labels = np.hstack((cat_labels, dog_labels))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Visualize decision boundary
def plot_decision_boundary(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(svm_classifier, X_test, y_test)
