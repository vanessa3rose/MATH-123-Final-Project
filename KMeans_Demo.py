import numpy as np
import matplotlib.pyplot as plt

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

# Plot the data
plt.figure(figsize=(8, 5))
plt.scatter(data[:, 0], data[:, 1], c='gray', marker='o', alpha=0.75)
plt.title("Original Data")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()


##################### RUN KMEANS CLUSTERING #####################

k = 2
max_iterations=4

# Initialize centroids as random data points
np.random.seed()
centroids = data[np.random.choice(data.shape[0], k, replace=False)]
colors = ['tab:red', 'tab:blue']

# Calculate the number of rows and columns for the subplots grid
num_plots = min(max_iterations, len(data))
num_rows, num_cols = 2, 2
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 8))
axs = axs.flatten()

# Iterate until convergence or max iterations reached
for iteration in range(max_iterations):
    # Calculate distances between each data point and each centroid
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    # Assign each data point to the closest centroid
    clusters = np.argmin(distances, axis=0)

    # Plot clusters with centroids
    axs[iteration].set_title(f"Iteration {iteration+1}")
    for i, centroid in enumerate(centroids):
        axs[iteration].scatter(data[clusters == i, 0], data[clusters == i, 1], c=colors[i], marker='o', alpha=0.75)
        # Label centroids
        axs[iteration].text(centroid[0], centroid[1], f"{['R', 'B'][i]}", fontsize=17, color='white',
                            weight='bold', horizontalalignment='center', verticalalignment='center')
        axs[iteration].text(centroid[0], centroid[1], f"{['R', 'B'][i]}", fontsize=14, color='black',
                            weight='bold', horizontalalignment='center', verticalalignment='center')
        
    axs[iteration].set_xticks([])
    axs[iteration].set_yticks([])

    # Update centroids based on mean of points in each cluster
    new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
    # Check for convergence
    if np.array_equal(centroids, new_centroids):
        break
    centroids = new_centroids

# Hide empty subplots
for i in range(iteration + 1, num_plots):
    fig.delaxes(axs[i])
    fig.add_subplot

# Set main title for all subplots
fig.suptitle("K-MEANS CLUSTERING",
             fontsize=15, y=0.975, color="darkkhaki", fontweight='bold', style='oblique')

plt.tight_layout()
plt.show()