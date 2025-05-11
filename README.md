# DecisionTree
###  **Iris Dataset – Overview**

The Iris dataset contains **150 samples** of iris flowers, with 3 species (classes):

* **Setosa**
* **Versicolor**
* **Virginica**

Each flower sample has **4 features**:

1. **Sepal length (cm)**
2. **Sepal width (cm)**
3. **Petal length (cm)**
4. **Petal width (cm)**

**Goal**: Classify an iris flower into one of the 3 species based on its measurements.

---

### **How Decision Tree works on Iris Dataset**

1️**Input**: Features (sepal & petal measurements)

2️**Splitting**:
The Decision Tree looks for a feature and threshold that best splits the dataset. For example:

```
Is petal length <= 2.45 cm?
├── Yes → Setosa
└── No
    ├── Is petal width <= 1.75 cm?
    │    ├── Yes → Versicolor
    │    └── No → Virginica
```

3 **Splitting Criteria**:
It uses **Gini Impurity** or **Entropy** to choose splits that best separate the classes.

4️**Output**:
The model predicts the species based on feature thresholds.

---

### **Advantages of Decision Tree on Iris Dataset**

* Simple and interpretable.
* Works well because the Iris dataset has clear, well-separated classes.
* No need to scale features.

---

###  **Limitations**

* A **small tree** works very well here because the classes are separable — but on more complex datasets, trees can **overfit**.

---

### **Basic Implementation (scikit-learn)**

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))


