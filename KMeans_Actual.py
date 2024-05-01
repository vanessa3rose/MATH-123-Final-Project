import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the data
data = pd.read_excel('PROJ/code/S&P_Stock_data.xlsx')

# Obtain average volatility and avg monthly return for each stock
avg_data = data[['Average Monthly Return', 'Beta (Volatility)']]
avg_data['Average Monthly Return'] *= 100

# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i in range(3):
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=i).fit(avg_data)
    clusters = kmeans.predict(avg_data)

    # Identify the index of the topmost point
    topmost_index = avg_data['Beta (Volatility)'].idxmax()

    # Ensures the topmost cluster will always be blue
    if clusters[avg_data.index == topmost_index][0] != 0:
        clusters = 1 - clusters

    # Define colors for each cluster
    colors = ['tab:blue' if label == 0 else 'tab:red' for label in clusters]

    # Plotting the data
    ax = axes[i]
    ax.scatter(avg_data['Average Monthly Return'], avg_data['Beta (Volatility)'], c=colors, alpha=0.75)
    ax.set_title(f'Iteration {i+1}')
    
    if i == 0:
        ax.set_ylabel('Beta (Volatility)')
    if i == 1:
        ax.set_xlabel('Average Monthly Return (%)')

# Set main title for all subplots
fig.suptitle("K-MEANS CLUSTERING",
             fontsize=15, y=0.975, color="darkkhaki", fontweight='bold', style='oblique')

plt.tight_layout(rect=[0.01, 0.01, 1, 1])
plt.show()