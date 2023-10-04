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
pixels = np.random.randint(0,25,10)
noise = np.random.normal(0, 0.5, 10)

S3 = S1
for i in range(10):
    S3[pixels[i]] = S1[pixels[i]] + noise[i]
    S3 = np.clip(S3, 0, 1)

J3 = J1
for i in range(10):
    J3[pixels[i]] = J1[pixels[i]] + noise[i]
    J3 = np.clip(J3, 0, 1)

H3 = H1
for i in range(10):
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


def normalise(mat):
    return (10 * (mat / np.linalg.norm(mat))).round(2)


# Create normalised combination matrices
SN = normalise(S1 + S2 + S3 + S4)
JN = normalise(J1 + J2 + J3 + J4)
HN = normalise(H1 + H2 + H3 + H4)

# Recognition matrix
NN1 = np.array([SN, JN, HN])


def recognise(nn, letter, name):

    plt.matshow(np.reshape(np.array(letter), [5, 5]), cmap="Greens")
    plt.show()

    # clarify 0s with negative weights
    for ii in range(len(letter)):
        if letter[ii] == 0:
            letter[ii] = letter[ii] - 0.1

    mat = np.matmul(nn, letter)
    normal = normalise(mat)
    amax = np.argmax(normal)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([0, 1, 2], normal, color=["#74baf7", "#fad56e", "#de6ef0"], width=0.6)
    ax.set_xticks(np.arange(0, 3, 1))
    ax.set_xticklabels(["S", "J", "H"], fontsize=20, fontfamily='serif')

    plt.title(name, fontsize=25, pad=15, fontfamily='serif')
    plt.savefig("plots/letter_recognition_" + name + ".png")
    plt.show()

    if amax == 0:
        return "S"
    elif amax == 1:
        return "J"
    elif amax == 2:
        return "H"


# Checks and plots all 12 letters
for i, l in enumerate([S1, S2, S3, S4, J1, J2, J3, J4, H1, H2, H3, H4]):
    name = "S" + str(i + 1)
    if i >= 4:
        name = "J" + str(i - 3)
    if i >= 8:
        name = "H" + str(i - 7)
    print(i + 1, recognise(NN1, l, name))


# Plot blurring matrix
plt.matshow(blurring_matrix, cmap="Greens")
plt.show()