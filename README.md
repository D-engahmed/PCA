# Principal Component Analysis (PCA)

## 1) PCA Concept

### A. Mathematical Idea

PCA (**Principal Component Analysis**) is a dimensionality reduction technique that:

***Transforms*** data into a new coordinate system where axes (principal components) capture the **maximum variance** in the data.
* The **first component** captures the largest variance, the second the next largest **orthogonal** variance, and so on.

---

#### **Step-by-Step Math**

Let’s assume we have a dataset $X$ with $n$ samples and $p$ features.

**1. Standardize Data**
We subtract the mean and divide by the standard deviation:

$$
Z_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}
$$

This is important because PCA is affected by scale.

**2. Covariance Matrix**
We compute:

$$
\Sigma = \frac{1}{n-1} Z^T Z
$$

This matrix shows how features vary together.

**3. Eigen Decomposition**
We solve:

$$
\Sigma v_k = \lambda_k v_k
$$

Where:

* $v_k$ = eigenvector (principal component direction)
* $\lambda_k$ = eigenvalue (variance explained by that component)

**4. Sort Components**
We order eigenvectors by descending eigenvalues.

**5. Project Data**
We transform data to the new coordinate system:

$$
Y = Z V_k
$$

Where $V_k$ contains the top $k$ eigenvectors.

---

### **B. Visualization**

The blue cloud represents the original 2D data. The **red arrows** show the **principal components**:

* **PC1** = the long axis (max variance direction).
* **PC2** = the short axis (remaining variance).

If we drop PC2, we reduce dimensions while keeping most of the information.

### **C. Interpretation**

* **PC1** is the direction of maximum variance.
* **PC2** is the direction of remaining variance.
* **PC1** explains most of the variance, so it’s the most important direction.
* **PC2** explains less variance, so it’s less important.

### **D. Application**

* **PCA** is used for **dimensionality reduction**.
* **PCA** is used for **data visualization**.
* **PCA** is used for **feature extraction**.
* **PCA** is used for **noise reduction**.

## **2) Code — Selecting the Number of Components in PCA**

We can use the **explained variance ratio** and **cumulative variance** to decide.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load example data
data = load_iris()
X = data.data

# 1. Standardize data
X_std = StandardScaler().fit_transform(X)

# 2. Apply PCA without specifying components
pca = PCA()
pca.fit(X_std)

# 3. Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
```

**Output:** The code standardizes the Iris dataset, applies PCA, and computes the explained variance ratio and cumulative variance for each principal component.

**Visualization:** The cumulative variance plot helps determine the optimal number of components to retain (e.g., where cumulative variance reaches ~95%).
