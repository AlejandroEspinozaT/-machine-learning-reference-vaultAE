#import "lib/template.typ": main
#import "lib/simpleTable.typ": simpleTable
#import "lib/codeBlock.typ": codeBlock

#show: doc => main(
  title: [
    Machine Learning Notes
  ],
  version: "v0.2.",
  authors: (
    (name: "Alejandro Espinoza T0rrico", email: "Alejandro.Espinoza@fundacion-jala.com"),
  ),
  abstract: [
    Este documento contiene apuntes resumidos y actualizados de las lecciones sobre Machine Learning,
    incluyendo conceptos de aprendizaje supervisado, no supervisado, regresión lineal y descenso por gradiente.
  ],
  doc,
)=

# Heading principal para la Semana
= Week 1: Introduction to Machine Learning

# Primera sección (antes era ##, ahora lo pones como ==)
== Video 1: Aprendizaje automático supervisado frente a no supervisado

- **What is Machine Learning?**
  - Definition attributed to **Arthur Samuel**: "Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed."
  - Example of **Samuel's Checkers Program**:
    - Samuel created a program that learned to play checkers by playing thousands of games against itself.
    - Through this experience, it identified patterns in board positions associated with winning or losing, improving its performance.
    - Outcome: The program eventually became a better player than Samuel himself.

- **Importance of Training Data**
  - The more data or experience a learning algorithm has, the better it can perform.
  - Practice quizzes are provided throughout the course to help reinforce understanding of these concepts.

- **Types of Machine Learning**
  - **Supervised Learning**: Most commonly used in real-world applications and the primary focus of this course’s first two segments.

  - **Unsupervised Learning**: Introduced in the third segment, along with recommender systems and reinforcement learning.

- **Practical Advice on Machine Learning**:
  - Many experienced teams often apply algorithms incorrectly, leading to inefficiencies.
  - This course aims to equip learners with both the tools and the know-how to avoid common pitfalls.

== Video 2: Supervised Learning Part 1

- **Importance of Supervised Learning**:
  - 99% of the current economic value created by machine learning comes from supervised learning.
  - Supervised learning algorithms learn mappings from **input (x)** to **output (y)**.
  
- **Definition of Supervised Learning**:
  - In supervised learning, algorithms are given examples with correct labels (input-output pairs).
  - The algorithm learns from these examples to predict output for new inputs.
  
- **Examples of Supervised Learning Applications**:
  - **Spam filtering**: Input = email, Output = spam or not spam.
  - **Speech recognition**: Input = audio clip, Output = text transcript.
  - **Machine translation**: Input = English text, Output = translation in another language.
  - **Online advertising**: Predicts likelihood of user clicking an ad (revenue-driven).
  - **Self-driving cars**: Input = image/sensor data, Output = location of other cars.
  - **Visual inspection in manufacturing**: Detects defects in products like scratches or dents.
- **Regression Example: Predicting Housing Prices**:
  - **Scenario**: Predict house prices based on the size of the house.
  - **Data Visualization**: Plot of house size (square feet) vs. price (thousands of dollars).
  - **Example predict
== Video 3: Supervised Learning Part 2

- **Types of Supervised Learning Algorithms**:
  - Supervised learning algorithms learn mappings from input (x) to output (y).
  - Two main types: **Regression** (predicts continuous numbers) and **Classification** (predicts categories).

- **Classification Algorithms**:
  - **Definition**: Classification algorithms predict a limited, finite set of possible categories or classes.
  - **Example - Breast Cancer Detection**:
    - Goal: Use a diagnostic tool to classify a tumor as benign (0) or malignant (1).
    - Data: Tumor size is plotted on a graph where the output is either 0 (benign) or 1 (malignant).
    - Difference from regression: Predicts discrete categories (e.g., 0 or 1) rather than continuous values.
  - **Multiple Categories**:
    - Classification can include more than two categories, e.g., different cancer types.
    - Terms "class" and "category" are used interchangeably.
    
- **Distinguishing Classification from Regression**:
  - Classification predicts a small set of categories (e.g., 0, 1, 2).
  - Unlike regression, classification doesn’t predict intermediate values (e.g., 0.5 or 1.7).

- **Using Multiple Inputs**:
  - **Example**: Tumor classification using both age and tumor size.
  - New dataset with two inputs (age and tumor size) to predict tumor type.
  - **Boundary in Classification**:
    - The learning algorithm may find a boundary (line) that separates categories (benign vs. malignant).
    - Additional inputs (e.g., cell size, cell shape) can improve predictions.
== Video 3: Unsupervised Learning Part 1
- **Introduction to Unsupervised Learning**:
  - Unsupervised learning is widely used after supervised learning.
  - It involves working with data **without output labels** (no "right answer" or supervision).
  - The goal is to **find patterns, structures, or clusters** in the data without predefined labels.

- **Clustering**:
  - **Definition**: Clustering is a type of unsupervised learning algorithm that groups data into clusters based on similarities.
  - Example - **Breast Cancer Data**:
    - With data on tumor size and patient age but no labels (benign/malignant), clustering can group the data based on similarities.

- **Applications of Clustering**:
  - **News Aggregation**:
    - Clustering is used to group related news stories (e.g., Google News groups articles about a single topic, like “panda birth”).
    - The algorithm identifies clusters based on shared keywords (e.g., “panda,” “twin,” “zoo”) without human supervision.
  - **Genetic Data Analysis**:
    - DNA microarray data represents genetic expression (rows = genes, columns = individuals).
    - Clustering groups individuals into types based on genetic similarities (e.g., “Type 1,” “Type 2”) without predefined categories.
  - **Market Segmentation**:
    - Companies use clustering on customer data to identify market segments.
    - Example from deep learning.ai: clustering identified groups with distinct motivations, such as skill growth, career advancement, and staying updated.

== Video 3:Unsupervised Learning Part 2
- **Formal Definition of Unsupervised Learning**:
  - In unsupervised learning, data consists only of **inputs (x)**, with **no output labels (y)**.
  - The goal is to **identify structure or patterns** within the data without predefined labels.

- **Types of Unsupervised Learning**:
  - **Clustering**:
    - Groups similar data points together into clusters.
  - **Anomaly Detection**:
    - Detects unusual events or data points that deviate from the norm.
    - Commonly used in **fraud detection** to identify suspicious financial transactions.
  - **Dimensionality Reduction**:
    - Reduces the size of a dataset by compressing data with minimal loss of information.
    - Useful for handling large datasets and reducing computational complexity.

- **Examples of Supervised vs. Unsupervised Learning**:
  - **Spam Filtering**: A **supervised learning** problem, as it uses labeled data (spam or non-spam).
  - **News Article Clustering**: An **unsupervised learning** problem, where clustering algorithms group similar articles without labels.
  - **Market Segmentation**: An **unsupervised learning** problem that discovers customer segments from unlabeled data.
  - **Diagnosing Diabetes**: A **supervised learning** problem, similar to the breast cancer example, using labeled data (diabetes or no diabetes).

...

= Week 1: Linear Regression Model

== Video 1: Linear Regression Model Part 1


- **Introduction to Linear Regression**:
  - Linear regression is a supervised learning algorithm that fits a straight line to data, commonly used for predicting numerical values (regression problems).
  - **Example**: Predicting the price of a house based on its size using a dataset of house sizes and prices from Portland, USA.
  
- **Supervised Learning**:
  - A supervised learning model is trained with labeled data (input-output pairs).
  - For house price prediction, inputs (x) are house sizes, and outputs (y) are house prices.

- **Regression vs. Classification**:
  - **Regression**: Predicts continuous values (e.g., house prices).
  - **Classification**: Predicts categories or discrete outputs (e.g., cat vs. dog).

- **Dataset Representation**:
  - Each data point is represented as an input feature (size of house) and target variable (price).
  - **Notation**:
    - **x**: Input feature (e.g., size of house).
    - **y**: Target output (e.g., price of house).
    - **m**: Total number of examples in the dataset (e.g., 47 houses).
    - **(x, y)**: Each pair represents a training example.
    - **xᵢ** and **yᵢ**: Represents the ith training example.

- **Training Set**:
  - The dataset used to train a model is called a **training set**.
  - **Goal**: Use the training set to create a model that can predict outputs for new inputs.
...

== Video 2: Linear Regression Model Part 2

- **Supervised Learning Process**:
  - A supervised learning algorithm takes a training set (input features and output targets) and produces a **function (f)** that predicts outputs for new inputs.
  - **y-hat (ŷ)**: Represents the model's prediction for the target variable y.

- **Linear Function**:
  - The model’s function (f) is represented as a linear equation:
    - **f(x) = wx + b**
    - **w**: Weight (slope of the line).
    - **b**: Bias (intercept).

- **Linear Regression with One Variable**:
  - This model is also known as **univariate linear regression** (one input feature).
  - **Goal**: Find the values of **w** and **b** that produce the best-fit line to minimize prediction error.

- **Cost Function**:
  - The cost function measures the error between predicted and actual values.
  - Minimizing the cost function is a key step in training the model and is widely used across machine learning.
...

== Video 3: Cost Function Formula

- **Introduction to the Cost Function**:
  - The **cost function** measures how well the linear regression model fits the training data.
  - **Goal**: Find the values of parameters **w** (weight) and **b** (bias) that minimize this cost function.

- **Parameters**:
  - **w** and **b** are the model parameters, determining the slope and y-intercept of the line.
  - Different values of **w** and **b** result in different lines, and the goal is to find values that minimize prediction error.

- **Defining the Cost Function**:
  - For each training example **i**, the model makes a prediction **ŷᵢ = f(xᵢ) = wxᵢ + b**.
  - The **error** for each example is the difference between the predicted value (ŷᵢ) and the actual target value (yᵢ).
  - The **squared error** for each example is **(ŷᵢ - yᵢ)²**.
  - The **cost function J(w, b)** is the average of the squared errors across all training examples:
    - **J(w, b) = (1 / 2m) ∑ (ŷᵢ - yᵢ)²** (where **m** is the number of examples).
    - The division by 2 is a convention to simplify calculations in optimization.

- **Goal of Linear Regression**:
  - Minimize the cost function **J(w, b)** to find the best values for **w** and **b**.

...

== Video 4: Cost Function Intuition

- **Understanding the Cost Function**:
  - The cost function **J(w, b)** measures how well the model's line fits the data.
  - **Goal**: Find **w** and **b** that minimize **J(w, b)**, resulting in the best-fit line.

- **Visualizing the Cost Function**:
  - Consider a simplified case with only **w** (setting **b** = 0):
    - For a given value of **w**, compute the line’s fit to the data and its cost **J(w)**.
    - Each value of **w** corresponds to a different line and a specific cost.
    - **J(w)** can be visualized as a curve where the minimum point represents the best-fit line.
  - Example:
    - If **w = 1**, the line perfectly fits example points (1, 1), (2, 2), and (3, 3), resulting in **J(1) = 0**.
    - If **w = 0.5** or **w = 0**, the cost **J** is higher, indicating a less accurate fit.

- **Minimizing the Cost**:
  - To find the optimal **w** and **b** values, select those that result in the smallest **J(w, b)**.
  - The best-fit line minimizes the squared errors, producing the lowest possible cost function value.

...

== Video 5: Cost Function Visualization
- **3D Visualization of Cost Function**:
  - In linear regression, the cost function **J(w, b)** resembles a 3D "bowl" shape.
  - **Goal**: Minimize **J(w, b)** by finding values of **w** and **b** that produce the best-fit line.
  - Each point on the surface represents a specific pair of values for **w** and **b**, with the height representing the cost.
  - The lowest point on this surface represents the minimum cost, where the model best fits the data.

- **Contour Plot**:
  - A **contour plot** represents the cost function in 2D by slicing horizontally through the "bowl."
  - Each ellipse represents points where the cost function **J(w, b)** has the same value.
  - The smallest ellipse at the center corresponds to the minimum cost, where **J(w, b)** is minimized.

...

== Video 6: Visualization Examples

- **Examples of Parameter Choices**:
  - Visualizations show how different values of **w** and **b** affect the line fit to the data and the cost **J(w, b)**.
  - **Example 1**: **w = -0.15**, **b = 800**.
    - The line has a high cost because it poorly fits the data.
  - **Example 2**: **w = 0**, **b = 360**.
    - Flat line (slope = 0), which is still a poor fit for the data, resulting in a relatively high cost.
  - **Example 3**: Better values of **w** and **b** lead to a line closer to the data points, minimizing **J(w, b)**.

- **Interactive Lab**:
  - In the optional lab, you can adjust **w** and **b** values to see how the line fit and cost **J(w, b)** change.
  - Includes interactive contour and 3D surface plots for hands-on exploration of the cost function.

= Week 1: Implementing Gradient Descent

== Video 1: Gradient Descent Overview

- **Gradient Descent**:
  - Gradient descent is an optimization algorithm used to minimize the cost function **J(w, b)** by iteratively adjusting parameters **w** and **b**.
  - This approach is used widely across machine learning, from simple linear regression to complex deep learning models.

- **Process**:
  - Start with initial guesses for **w** and **b** (often **w = 0** and **b = 0**).
  - Adjust **w** and **b** in small steps to reduce **J(w, b)**, iterating until the cost stabilizes at a minimum.

- **Intuition**:
  - Imagine standing on a hill and taking small steps downhill (steepest descent) to reach the lowest point, or local minimum.


== Video 2: Implementing Gradient Descent

- **Update Rule**:
  - For each iteration, update **w** and **b** as follows:
    - **w = w - α * (dJ/dw)**
    - **b = b - α * (dJ/db)**
    - **α** (alpha) is the **learning rate**, controlling the size of each step.
  - **Simultaneous Update**: Both **w** and **b** should be updated at the same time to ensure correct implementation.

- **Learning Rate (α)**:
  - A small **α** results in slow progress, while a large **α** can cause overshooting and failure to converge.
  - The choice of **α** greatly affects the efficiency and success of gradient descent.


...
== Video 3: Gradient Descent Intuition
- **Derivative Term**:
  - The derivative **dJ/dw** (or **dJ/db**) indicates the direction to move **w** (or **b**) to reduce **J**.
  - Moving in the direction opposite to the derivative (steepest descent) leads towards the minimum.

- **Effect of Learning Rate**:
  - If **α** is too small, the algorithm converges slowly.
  - If **α** is too large, the algorithm may diverge by overshooting the minimum repeatedly.
== Video 4: Learning Rate Details
- **Choosing the Learning Rate**:
  - A moderate learning rate results in a balance between convergence speed and stability.
  - If the parameters reach a local minimum, further updates stop since the derivative becomes zero.

== Video 5: Gradient Descent for Linear Regression

- **Applying Gradient Descent to Linear Regression**:
  - With **J(w, b)** as the mean squared error cost function, compute the derivatives:
    - **dJ/dw = (1/m) Σ (ŷᵢ - yᵢ) * xᵢ**
    - **dJ/db = (1/m) Σ (ŷᵢ - yᵢ)**
  - The derivatives allow **w** and **b** to be updated at each step.

- **Convexity**:
  - The cost function for linear regression is convex (bowl-shaped), meaning there is only one global minimum, ensuring convergence.


== Video 6: Running Gradient Descent

- **Visualization**:
  - Visualize the linear regression line fit improving with each iteration as **J(w, b)** moves closer to the global minimum.
  - After convergence, the final line fit can be used to make predictions, such as predicting a house price based on its size.

- **Batch Gradient Descent**:
  - This version of gradient descent uses the entire training set at each update step, ensuring stability for small datasets.
  - Other gradient descent variants exist that use subsets of data, known as **stochastic** or **mini-batch gradient descent**.

- **Summary**:
  - Successfully applying gradient descent to linear regression completes the first machine learning model, preparing you to handle more complex models in the future.

= Week 2: Multiple Linear Regression

== Video 1: Multiple Features
- Multiple linear regression extends linear regression to multiple features (e.g., size, number of bedrooms).
- Notation:
  - \(x_j\): j-th feature.
  - Model: \(f_{w,b}(x) = w_1x_1 + w_2x_2 + ... + w_nx_n + b\).
- Using vectors for parameters (\(\mathbf{w}\)) and inputs (\(\mathbf{x}\)) simplifies representation.
- "Multiple linear regression" uses multiple features to improve predictions.

== Video 2: Vectorization Part 1
- **Vectorization**: a technique for operating on vectors and matrices efficiently.
- Replaces loops with vectorized operations (e.g., `np.dot(w, x) + b`).
- Advantages: shorter, faster, and scalable code.

== Video 3: Vectorization Part 2
- With vectorization, operations run in parallel instead of sequentially.
- Speeds up calculations like parameter updates in gradient descent.
- Especially beneficial for large datasets.
- Practical lab: compare runtimes between vectorized and non-vectorized code.

== Video 4: Gradient Descent for Multiple Linear Regression
- Gradient descent is applied to the multivariable case.
- Updates for \(w_j\) and \(b\) are similar to the univariate case.
- The cost \(J(w,b)\) remains convex, ensuring a global minimum.
- Alternative: Normal equation (less practical for large \(n\), but available).

== Summary
- Multiple linear regression incorporates multiple features for better predictions.
- Vectorization makes code more efficient.
- These techniques lay the groundwork for more complex problems.

= Week 2: Gradient Descent in Practice

== Video 1: Feature Scaling Part 1
- **Feature scaling**: Normalizing features accelerates gradient descent.
- Without scaling, contour plots are elongated, slowing the process.
- Normalizing makes contours more "circular," improving convergence speed.

== Video 2: Feature Scaling Part 2
- Methods: rescaling (dividing by range), mean normalization, z-score.
- Goal: center features around 0 with similar ranges.
- Enhances convergence and stabilizes training.

== Video 3: Checking Gradient Descent for Convergence
- Monitor the cost \(J\) over iterations.
- If \(J\) stops decreasing, the model converges.
- If \(J\) increases, review the learning rate (\(\alpha\)) or the implementation.

== Video 4: Choosing the Learning Rate
- Learning rate (\(\alpha\)) too small: slow convergence.
- \(\alpha\) too large: fails to converge or fluctuates.
- Test various values and choose the largest one that still ensures stable convergence.

== Video 5: Feature Engineering
- Create new features from existing ones to better capture relationships.
- Example: area = width × length.
- Can improve accuracy and model capacity to fit complex patterns.

== Video 6: Polynomial Regression
- Add powers (\(x^2, x^3\)) or other transformations.
- Enables fitting non-linear relationships.
- Scaling generated features (e.g., \(x^2, x^3\)) is crucial due to their large range.
- Using libraries (e.g., scikit-learn) simplifies implementation.
- Prepares for the transition from regression to classification in upcoming weeks.
= Week 3: Regression and Classification

== Video 1: Motivations
- This week introduces classification.
- Linear regression is unsuitable for classification problems.
- In classification, \(y\) takes on categorical values, often 0 or 1.
- Examples: spam/no spam, benign/malignant tumor.
- Using linear regression for classification can result in unstable decision boundaries.
- Logistic regression solves these issues by keeping outputs between 0 and 1 and avoiding linear regression's limitations.

== Video 2: Logistic Regression
- Logistic regression is one of the most widely used binary classification algorithms.
- It uses the sigmoid function: \( f(x) = \frac{1}{1+e^{-z}} \), where \( z = w \cdot x + b \).
- The output is a probability between 0 and 1.
- If \( f(x) \geq 0.5 \), predict \(y=1\); otherwise, \(y=0\).

== Video 3: Decision Boundary
- The decision boundary is established where \( z = w \cdot x + b = 0 \).
- With linear features, the decision boundary is a line (2D) or a plane (3D).
- Polynomial features allow for non-linear decision boundaries, enabling better fits for complex data.

= Week 3: Cost Function for Logistic Regression

== Video 1: Cost Function for Logistic Regression
- Logistic regression does not use the squared error cost function.
- Instead, it uses the logarithmic loss function:
  - If \(y=1\): \(L = -\log(f(x))\).
  - If \(y=0\): \(L = -\log(1-f(x))\).
- The total cost \(J(w,b)\) is the mean of all losses.
- This function is convex, ensuring reliable gradient descent convergence.

== Video 2: Simplified Cost Function
- The unified cost expression is:
  \[
    J(w,b) = -\frac{1}{m}\sum_{i=1}^m [y^{(i)}\log(f(x^{(i)})) + (1-y^{(i)})\log(1-f(x^{(i)}))]
  \]
- This cost function is derived from maximum likelihood estimation and is convex.

= Week 3: Gradient Descent Implementation

== Video 1: Gradient Descent Implementation
- To find optimal \(w, b\), gradient descent is used.
- Updates:
  - \( w_j := w_j - \alpha \frac{1}{m}\sum (f(x^{(i)})-y^{(i)})x_j^{(i)} \)
  - \( b := b - \alpha \frac{1}{m}\sum (f(x^{(i)})-y^{(i)}) \)
- The key difference from linear regression is the use of the sigmoid function in \(f(x)\).
- Vectorization and feature scaling improve convergence speed.
- Optional labs: Implement logistic regression with gradient descent, visualizations, and scikit-learn integration.

= Week 3: The Problem of Overfitting

== Video 1: The Problem of Overfitting
- Overfitting: The model fits the training data too closely and fails to generalize.
- Underfitting: The model fails to capture patterns; it is too simple.
- Examples:
  - Underfitting: A simple line on complex data.
  - Overfitting: High-degree polynomial that fits noise and fails to generalize.
- Applies to both regression and classification.

== Video 2: Addressing Overfitting
- Strategies:
  1. Collect more data.
  2. Feature selection: Reduce the number of features.
  3. Regularization: Penalize large parameter values without removing features.
- Regularization balances complexity and fit.

== Video 3: Cost Function with Regularization
- Regularization adds a term \(\frac{\lambda}{2m}\sum w_j^2\) to the cost function.
- This controls the size of \(w_j\) to prevent overly complex curves.
- Adjusting \(\lambda\) balances training fit and model simplicity.

== Video 4: Regularized Linear Regression
- Gradient descent with regularization:
  - The gradient adds \(\frac{\lambda}{m}w_j\) to the derivative term.
  - Prevents \(w_j\) from growing too large.
- Enables the use of all features without overfitting.

== Video 5: Regularized Logistic Regression
- Regularization also applies to logistic regression.
- Cost function:
  \[
    J(w,b)= -\frac{1}{m}\sum [y\log(f(x))+(1-y)\log(1-f(x))] + \frac{\lambda}{2m}\sum w_j^2
  \]
- Prevents overly complex decision boundaries even with high-order polynomial features.
- Gradient descent updates include a penalty term for \(w_j\), excluding \(b\).

= Conclusion of the Course
- This concludes the first part of the specialization.
- You’ve learned linear regression, logistic regression, gradient descent, vectorization, and how to handle overfitting.
- Next steps: Neural networks and deep learning in future courses.

