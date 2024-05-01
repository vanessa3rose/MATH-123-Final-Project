import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

######################### PREPARE DATA #########################
np.random.seed(0)

# Generate random data for the first cluster at y around 0.9
x1 = np.random.uniform(0.5, 1.0, 100)
y1 = np.random.uniform(0.55, 1.0, 100)

# Generate random data for the second cluster at y around 0.1
x2 = np.random.uniform(0.0, 0.5, 100)
y2 = np.random.uniform(0.0, 0.45, 100)

# Combine the data from both clusters
cluster1 = np.column_stack((x1, y1))
cluster2 = np.column_stack((x2, y2))
data = np.concatenate([cluster1, cluster2])

# Modify the original data by changing its labels to be either TRUE (1) or FALSE (0)
data_modified = data.copy()
data_modified[:, 1] = np.where(data_modified[:, 1] > 0.5, 1, 0)

# Create a plot for the data
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot the original data
axes[0].scatter(data[:, 0], data[:, 1], c='gray', marker='o', alpha=0.75)
axes[0].set_title("Original Data")
axes[0].set_xlim(-0.1, 1.1)
axes[0].set_ylim(-0.1, 1.1)

# Plot the modified data
axes[1].scatter(data_modified[:, 0], data_modified[:, 1], c='gray', marker='o', alpha=0.75)
axes[1].set_title("Simplified Data")
axes[1].set_xlim(-0.1, 1.1)
axes[1].set_ylim(-0.1, 1.1)

# Layout adjustment
plt.tight_layout()
plt.show()


#################### RUN LOGISTIC REGRESSION ####################

X = data_modified[:, [0]]  # feature data
y = data_modified[:, 1]    # labels

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model on the test data
y_pred = model.predict(X_test)

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

# Determine colors based on the labels
colors_train = np.where(y_train == 1, 'tab:blue', 'tab:red')
colors_test = np.where(y_test == 1, 'tab:blue', 'tab:red')

# Subplot 1: Training data with logistic regression line
x_values = np.linspace(X_train.min(), X_train.max(), 300)
y_probs = model.predict_proba(x_values.reshape(-1, 1))[:, 1]
axs[0].plot(x_values, y_probs, color='darkkhaki', linewidth=2.5)

axs[0].scatter(X_train, y_train, color=colors_train, alpha=0.75)
axs[0].set_title('Traning Data')
axs[0].set_ylim(-0.2, 1.2)
axs[0].set_yticks([0, 1])
axs[2].set_yticklabels(['0', '1'])

# Subplot 2: Actual labels
axs[1].scatter(X_test, y_test, color=colors_test, alpha=0.75)
axs[1].set_title('Actual Labels of Test Data')
axs[1].set_ylim(-0.2, 1.2)
axs[1].set_yticks([0, 1])
axs[2].set_yticklabels(['0', '1'])

# Subplot 3: Predictions
axs[2].scatter(X_test, y_pred, color=colors_test, alpha=0.75)
axs[2].set_title('Model Predictions of Test Data Labels')
axs[2].set_ylim(-0.2, 1.2)
axs[2].set_yticks([0, 1])
axs[2].set_yticklabels(['0', '1'])

# Set main title for all subplots
fig.suptitle("LOGISTIC REGRESSION MODEL - {:.2f}% ACCURACY".format(model.score(X_test, y_test) * 100),
             fontsize=15, y=0.975, color="darkkhaki", fontweight='bold', style='oblique')

plt.tight_layout()
plt.show()