import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Basic characters
S = [0.99, 0.99, 0.99, 0.99, 0.99,
     0.99, 0, 0, 0, 0,
     0.99, 0.99, 0.99, 0.99, 0.99,
     0, 0, 0, 0, 0.99,
     0.99, 0.99, 0.99, 0.99, 0.99]

J = [0.99, 0.99, 0.99, 0.99, 0.99,
     0, 0, 0.99, 0, 0,
     0, 0, 0.99, 0, 0,
     0.99, 0, 0.99, 0, 0,
     0.99, 0.99, 0.99, 0, 0]

H = [0.99, 0, 0, 0, 0.99,
     0.99, 0, 0, 0, 0.99,
     0.99, 0.99, 0.99, 0.99, 0.99,
     0.99, 0, 0, 0, 0.99,
     0.99, 0, 0, 0, 0.99]

# Character variation 1 (emphasizing strokes)
S1 = [0.99, 0.99, 0.99, 0.7, 0.5,
      0.99, 0, 0, 0, 0,
      0.99, 0.99, 0.99, 0.99, 0.99,
      0, 0, 0, 0, 0.99,
      0.5, 0.7, 0.5, 0.6, 0.99]

J1 = [0.5, 0.9, 0.99, 0.8, 0.6,
      0, 0, 0.99, 0, 0,
      0, 0, 0.99, 0, 0,
      0.4, 0, 0.99, 0, 0,
      0.6, 0.8, 0.99, 0, 0]

H1 = [0.6, 0, 0, 0, 0.99,
      0.8, 0, 0, 0, 0.99,
      0.8, 0.99, 0.99, 0.99, 0.99,
      0.7, 0, 0, 0, 0.99,
      0.6, 0, 0, 0, 0.99]

# Character variation 2 (blur)
blurring_matrix = np.array(
    [
        [0.6, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.2, 0.4, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.2, 0.4, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.2, 0.4, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.2, 0.6, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.2, 0, 0, 0, 0, 0.4, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.4, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.4, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.4, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.4, 0.2, 0, 0, 0, 0.2, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0, 0.2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0, 0.2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0, 0.2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.4, 0, 0, 0, 0, 0.2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.6, 0.2, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.4, 0.2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.4, 0.2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.4, 0.2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0.2, 0.6]
    ]
)

S2 = blurring_matrix.dot(S1)
J2 = blurring_matrix.dot(J1)
H2 = blurring_matrix.dot(H1)

# Character variation 3 (noise)
np.random.seed(0)
pixels = np.random.randint(0,25,5)
noise = np.random.normal(0, 0.5, 5)

S3 = S1
for i in range(5):
    S3[pixels[i]] = S1[pixels[i]] + noise[i]
    S3 = np.clip(S3, 0, 1)

J3 = J1
for i in range(5):
    J3[pixels[i]] = J1[pixels[i]] + noise[i]
    J3 = np.clip(J3, 0, 1)

H3 = H1
for i in range(5):
    H3[pixels[i]] = H1[pixels[i]] + noise[i]
    H3 = np.clip(H3, 0, 1)

# Character variation 4 (blur + noise)
S4 = blurring_matrix.dot(S3)
J4 = blurring_matrix.dot(J3)
H4 = blurring_matrix.dot(H3)

# Correlation matrix
data = {'S1': S1,
        'S2': S2,
        'S3': S3,
        'S4': S4,
        'J1': J1,
        'J2': J2,
        'J3': J3,
        'J4': J4,
        'H1': H1,
        'H2': H2,
        'H3': H3,
        'H4': H4}

df = pd.DataFrame(data)
corr = df.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(260, 25, s=80, center="light", as_cmap=True)

# Draw the heatmap
ax = sns.heatmap(corr, ax=ax, linewidths=0.5, linecolor='black', clip_on=False, cmap=cmap)

# Add gridlines to separate each letter
ax.axhline(y=4, color='k', linewidth=5)
ax.axhline(y=8, color='k', linewidth=5)
ax.axvline(x=4, color='k', linewidth=5)
ax.axvline(x=8, color='k', linewidth=5)

# Adjust tick size and padding
plt.xticks(fontsize=15)
plt.yticks(fontsize=15, rotation=0)
ax.tick_params(axis='both', which='major', labelfontfamily='serif', pad=5)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15, labelfontfamily='serif', pad=5)

plt.title("Letter Correlation Matrix", fontsize=25, pad=20, fontfamily='serif')
plt.savefig("letter_correlation_matrix.png")
plt.show()


# Recognition matrix
NN1 = np.array([S, J, H])

# Plot chosen letter
test = H2
plt.matshow(np.reshape(np.array(test), [5,5]), cmap="Blues")
plt.show()

# Test NN
print(np.matmul(NN1, test))
print(np.argmax(np.matmul(NN1, test)))
plt.bar([0,1,2], np.matmul(NN1, test), tick_label=["S", "J", "H"], color=["#74baf7", "#fad56e", "#de6ef0"])
plt.show()


"""
print(str(np.dot(S, S)) + '\t' + str(np.dot(S, J)) + '\t' + str(np.dot(S, H)))
print(str(np.dot(J, S)) + '\t' + str(np.dot(J, J)) + '\t' + str(np.dot(J, H)))
print(str(np.dot(H, S)) + '\t' + str(np.dot(H, J)) + '\t' + str(np.dot(H, H)))
"""