# Week 3: Regression and Classification

## Video 1: Motivations

- **Introduction**:
  - Welcome to the third week of this course.
  - By the end of this week, you will complete the first course of this specialization.

- **Key Concept**:
  - Last week: Linear regression, which predicts a continuous number.
  - This week: Classification, where the output variable \(y\) takes on one of a small set of possible values.

- **Limitations of Linear Regression for Classification**:
  - Linear regression is not suitable for classification problems.
  - Classification requires predicting categories, not continuous values.

- **Binary Classification**:
  - Problems where \(y\) can be one of two values (e.g., No/Yes, False/True, or 0/1).
  - Examples:
    - Is an email spam? (No/Yes)
    - Is a financial transaction fraudulent? (No/Yes)
    - Is a tumor malignant or benign? (0/1)
  - Terminology:
    - **Negative Class**: Represented by 0 (e.g., not spam, benign).
    - **Positive Class**: Represented by 1 (e.g., spam, malignant).

- **Visualization**:
  - Training set example: Tumor classification (malignant = 1, benign = 0).
  - Tumor size plotted on the horizontal axis and label \(y\) on the vertical axis.

- **Linear Regression as a Classifier**:
  - Attempting to use linear regression to fit a straight line to the data:
    - Predicts values between 0 and 1.
    - Uses a threshold (e.g., 0.5) to classify data:
      - \(y = 0\) for outputs below 0.5.
      - \(y = 1\) for outputs 0.5 or above.

- **Challenges with Linear Regression**:
  - Adding new data points can shift the decision boundary undesirably.
  - Example:
    - Adding a single outlier changes the dividing line, leading to incorrect classifications.
  - Linear regression does not produce robust decision boundaries for classification tasks.

- **Introducing Logistic Regression**:
  - Addresses the limitations of linear regression for classification.
  - Output is always between 0 and 1.
  - Avoids the shifting decision boundary problem.
  - Note:
    - Despite its name, logistic regression is a classification algorithm.
    - Used for binary classification (0 or 1).


# Week 3: Regression and Classification

## Video 2: Logistic Regression

- **Introduction**:
  - Logistic regression is one of the most widely used classification algorithms.
  - It is effective for binary classification tasks (e.g., classifying tumors as malignant or benign).

- **Key Concepts**:
  - Logistic regression fits an S-shaped curve (sigmoid curve) to the data.
  - Output is a probability between 0 and 1, representing the likelihood of a class (e.g., malignant vs. benign).

- **Sigmoid Function**:
  - Formula: \( g(z) = \frac{1}{1 + e^{-z}} \), where \( e \) is approximately 2.718.
  - Properties:
    - \( g(z) \approx 1 \): when \( z \) is large and positive.
    - \( g(z) \approx 0 \): when \( z \) is large and negative.
    - \( g(0) = 0.5 \): passes through 0.5 on the vertical axis.

- **Logistic Regression Model**:
  - Step 1: Compute \( z = \mathbf{w} \cdot \mathbf{x} + b \), where:
    - \( \mathbf{w} \): weights.
    - \( b \): bias.
  - Step 2: Apply the sigmoid function: \( f(x) = g(z) = \frac{1}{1 + e^{-z}} \).
  - Outputs a value between 0 and 1.

- **Interpretation**:
  - \( f(x) \): Probability that \( y = 1 \) given input \( x \).
  - Example:
    - \( f(x) = 0.7 \): The model predicts a 70% chance that \( y = 1 \) (e.g., tumor is malignant).
    - Complementary probability: \( 1 - f(x) = 0.3 \) (30% chance that \( y = 0 \)).

- ** Notes**:
  - Logistic regression is effective for binary classification.
  - Widely used in fields such as Internet advertising to predict probabilities of outcomes.


## Video 3: Decision Boundary

- **Recap**:
  - Logistic regression outputs \( f(x) = g(z) \), where \( z = \mathbf{w} \cdot \mathbf{x} + b \).
  - The sigmoid function maps \( z \) to a value between 0 and 1.

- **Thresholding**:
  - Decision threshold (commonly 0.5):
    - \( f(x) \geq 0.5 \): Predict \( y = 1 \).
    - \( f(x) < 0.5 \): Predict \( y = 0 \).
  - \( z \geq 0 \): Corresponds to \( f(x) \geq 0.5 \).

- **Visualization**:
  - **Linear Decision Boundary**:
    - Example with two features (\( x_1, x_2 \)):
      - \( z = w_1x_1 + w_2x_2 + b \).
      - Decision boundary: \( z = 0 \) (e.g., \( x_1 + x_2 = 3 \)).
    - Region where \( z \geq 0 \): Predict \( y = 1 \).
    - Region where \( z < 0 \): Predict \( y = 0 \).
  - **Non-Linear Decision Boundary**:
    - Using polynomial features:
      - Example: \( z = w_1x_1^2 + w_2x_2^2 + b \).
      - Decision boundary (e.g., \( x_1^2 + x_2^2 = 1 \)) forms a circle.
    - More complex boundaries:
      - Higher-order terms create ellipses or other complex shapes.

- **Key Insights**:
  - Logistic regression produces linear decision boundaries with basic features.
  - Adding polynomial features allows for non-linear decision boundaries, enabling more complex classification tasks.


# Week 3: Cost Function for Logistic Regression

## Video 1: Cost Function for Logistic Regression

- **Overview**:
  - The cost function measures how well a specific set of parameters fits the training data.
  - For logistic regression, the squared error cost function is not suitable due to its non-convex nature.

- **Training Data**:
  - Number of examples: \( m \).
  - Features: \( X_1, X_2, ..., X_n \).
  - Target label \( y \): Binary values (0 or 1).

- **Challenges with Squared Error**:
  - Squared error cost function results in a non-convex surface for logistic regression.
  - Gradient descent may get stuck in local minima, making it unreliable.

- **New Loss Function**:
  - The loss function measures how well the model performs on a single training example:
    - \( L(f(x), y) = \begin{cases} 
      -\log(f(x)), & \text{if } y = 1, \\
      -\log(1 - f(x)), & \text{if } y = 0.
      \end{cases} \)
  - Properties:
    - If \( y = 1 \), the loss decreases as \( f(x) \) approaches 1.
    - If \( y = 0 \), the loss decreases as \( f(x) \) approaches 0.

- **Behavior of the Loss Function**:
  - \( y = 1 \): 
    - Loss is small if \( f(x) \approx 1 \).
    - Loss increases as \( f(x) \) moves away from 1.
  - \( y = 0 \):
    - Loss is small if \( f(x) \approx 0 \).
    - Loss increases as \( f(x) \) approaches 1.

- **Convex Cost Function**:
  - Using this loss function, the cost function becomes:
    - \( J(w, b) = \frac{1}{m} \sum_{i=1}^{m} L(f(x^{(i)}), y^{(i)}) \).
  - Convexity ensures gradient descent reliably converges to a global minimum.



## Video 2: Simplified Cost Function

- **Simplified Loss Function**:
  - Unified equation for the loss:
    - \( L(f(x), y) = -y \log(f(x)) - (1 - y) \log(1 - f(x)) \).
  - This single expression handles both \( y = 0 \) and \( y = 1 \) cases:
    - If \( y = 1 \): \( L = -\log(f(x)) \).
    - If \( y = 0 \): \( L = -\log(1 - f(x)) \).

- **Cost Function**:
  - Average loss over all training examples:
    - \( J(w, b) = \frac{1}{m} \sum_{i=1}^{m} [-y^{(i)} \log(f(x^{(i)})) - (1 - y^{(i)}) \log(1 - f(x^{(i)}))] \).
  - Properties:
    - Derived from maximum likelihood estimation (a statistical principle).
    - Convexity guarantees reliable convergence with gradient descent.

- **Why This Cost Function?**:
  - It is rooted in statistical principles, specifically maximum likelihood estimation.
  - Ensures a convex surface for optimization.



# Week 3: Gradient Descent Implementation

## Video 1: Gradient Descent Implementation

- **Overview**:
  - Goal: Minimize the cost function \( J(w, b) \) to find the optimal parameters \( w \) and \( b \) for logistic regression.
  - After fitting the model, it can predict the probability that \( y = 1 \) for a given input \( x \).

- **Gradient Descent Algorithm**:
  - Parameters \( w_j \) and \( b \) are updated iteratively:
    - \( w_j := w_j - \alpha \frac{\partial J}{\partial w_j} \)
    - \( b := b - \alpha \frac{\partial J}{\partial b} \)
  - \( \alpha \): Learning rate.
  - Updates are performed simultaneously for all parameters.

- **Gradient Derivatives**:
  - Derivative with respect to \( w_j \):
    - \( \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \big( f(x^{(i)}) - y^{(i)} \big) x_j^{(i)} \)
  - Derivative with respect to \( b \):
    - \( \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \big( f(x^{(i)}) - y^{(i)} \big) \)
  - \( f(x) = \sigma(w \cdot x + b) \), where \( \sigma \) is the sigmoid function.

- **Key Differences Between Linear and Logistic Regression**:
  - The equations for gradient descent look similar, but the definitions of \( f(x) \) differ:
    - Linear regression: \( f(x) = w \cdot x + b \).
    - Logistic regression: \( f(x) = \sigma(w \cdot x + b) \).
  - Logistic regression uses the sigmoid function, making it distinct from linear regression.

- **Vectorized Implementation**:
  - Gradient descent can be implemented using vectorization for faster computation.
  - The same techniques used for linear regression vectorization can be applied.

- **Feature Scaling**:
  - Scaling features to a similar range (e.g., \([-1, 1]\)) helps gradient descent converge faster.
  - Feature scaling is as beneficial for logistic regression as it is for linear regression.

- **Optional Labs**:
  - **Gradient Descent for Logistic Regression**:
    - Implement gradient descent in code.
    - Visualize:
      - Sigmoid function.
      - Contour plots of the cost function.
      - 3D surface plots of the cost function.
      - Learning curves.
  - **Scikit-learn for Logistic Regression**:
    - Learn to train logistic regression models using the popular Scikit-learn library.
    - Explore how this library is used in real-world machine learning workflows.

- **Key Takeaways**:
  - You now know how to implement logistic regression with gradient descent.
  - Logistic regression is a powerful and widely used algorithm in machine learning.

- **Next Steps**:
  - Apply these concepts in practice labs.
  - Visualize how gradient descent works and explore its applications using Scikit-learn.

# Week 3: The Problem of Overfitting

## Video 1: The Problem of Overfitting

- **Overview**:
  - Overfitting and underfitting are common issues in machine learning.
  - Overfitting occurs when a model fits the training data too well but fails to generalize.
  - Underfitting occurs when a model fails to capture the underlying pattern in the training data.

- **Examples of Overfitting and Underfitting**:
  - **Underfitting**:
    - Linear regression on housing prices with size as the feature.
    - The model assumes a straight line, leading to poor fit and high bias.
  - **Balanced Model**:
    - Quadratic regression with features \( x \) and \( x^2 \).
    - Fits the training data reasonably well and generalizes to new data.
  - **Overfitting**:
    - High-order polynomial regression (e.g., \( x, x^2, x^3, x^4 \)).
    - Fits the training data perfectly but produces wiggly curves and fails to generalize.
    - Known as high variance.

- **Key Terms**:
  - **Bias**:
    - High bias corresponds to underfitting.
    - The model has a strong assumption (e.g., data is linear) that prevents it from fitting the data well.
  - **Variance**:
    - High variance corresponds to overfitting.
    - The model captures noise in the training data and produces unstable predictions.

- **Overfitting in Classification**:
  - Example: Classifying tumors as malignant or benign with features \( x_1 \) (tumor size) and \( x_2 \) (patient age).
    - **Underfitting**:
      - Logistic regression with a simple linear decision boundary.
    - **Balanced Model**:
      - Logistic regression with quadratic features produces an elliptical decision boundary.
    - **Overfitting**:
      - High-order polynomial features produce overly complex decision boundaries that fail to generalize.



## Video 2: Addressing Overfitting

- **Techniques to Address Overfitting**:
  1. **Collect More Data**:
     - Larger training sets reduce overfitting by helping the model learn less wiggly functions.
     - May not always be feasible due to data availability constraints.
  2. **Feature Selection**:
     - Use a subset of the most relevant features (e.g., size, bedrooms, age).
     - Reduces overfitting by limiting the number of features.
     - Drawback: May discard useful information.
     - Algorithms for automatic feature selection are covered in later courses.
  3. **Regularization**:
     - Shrinks parameter values (\( w_1, w_2, ..., w_n \)) without eliminating features.
     - Prevents features from having an overly large influence on predictions.
     - Helps higher-order polynomial models fit the data better.

- **Regularization**:
  - Encourages smaller parameter values to reduce overfitting.
  - Typically applied to \( w_1, w_2, ..., w_n \); the bias term (\( b \)) is often excluded.
  - Allows retaining all features while controlling their impact.

- **Next Steps**:
  - Learn the mathematical formulation of regularization.
  - Apply regularization to linear regression, logistic regression, and other algorithms to reduce overfitting.


## Video 3: Cost Function with Regularization

- **Overview**:
  - Regularization helps reduce overfitting by shrinking parameter values (\(w_1, w_2, ..., w_n\)).
  - Modified cost function incorporates a regularization term to penalize large parameter values.

- **Example**:
  - Quadratic function provides a good fit to data.
  - High-order polynomial overfits the data, resulting in a wiggly curve.
  - Regularization shrinks parameters (e.g., \(w_3, w_4\)) to minimize their impact, producing a simpler and smoother model.

- **Regularization in Practice**:
  - Adds a penalty term to the cost function:
    - \( J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2 \)
  - \( \lambda \): Regularization parameter.
    - Controls the trade-off between fitting the training data and keeping parameters small.
  - By convention, the bias term (\(b\)) is not regularized.

- **Effect of Regularization**:
  - If \( \lambda = 0 \): No regularization; the model may overfit.
  - If \( \lambda \) is too large: Parameters shrink too much; the model underfits.
  - Ideal \( \lambda \): Balances minimizing the cost and reducing overfitting.

- **Key Insights**:
  - Regularization penalizes large parameter values, encouraging simpler models.
  - Adding regularization keeps all features but limits their influence.
  - Regularization term scales with \( \frac{1}{2m} \) to ensure consistency as the training set grows.


## Video 4: Regularized Linear Regression

- **Gradient Descent with Regularization**:
  - Updates for parameters:
    - \( w_j := w_j - \alpha \left( \frac{\partial J}{\partial w_j} \right) \), where:
      - \( \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \big( f(x^{(i)}) - y^{(i)} \big) x_j^{(i)} + \frac{\lambda}{m} w_j \)
    - \( b := b - \alpha \left( \frac{\partial J}{\partial b} \right) \), where:
      - \( \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \big( f(x^{(i)}) - y^{(i)} \big) \)
  - Regularization affects only \( w_j \), not \( b \).

- **Interpretation of Updates**:
  - Regularization shrinks \( w_j \) slightly in each iteration:
    - \( w_j := w_j \cdot (1 - \alpha \cdot \frac{\lambda}{m}) - \alpha \cdot \text{(gradient term)} \)
  - Multiplier \( 1 - \alpha \cdot \frac{\lambda}{m} \) reduces \( w_j \), ensuring smaller parameter values over time.

- **Impact of Regularization**:
  - Helps linear regression perform better when the training set is small or has many features.
  - Reduces overfitting by limiting parameter magnitudes.

- **Optional Details**:
  - Derivatives:
    - Regularization adds \( \frac{\lambda}{m} w_j \) to the gradient term.
    - Simplifies gradient descent without requiring changes to core algorithmic structure.
  - Mathematical Explanation:
    - Regularization shrinks parameters iteratively, preventing overfitting.

- **Key Takeaways**:
  - Regularized linear regression improves generalization by controlling overfitting.
  - Adjusting \( \lambda \) balances the fit between training data and parameter simplicity.
## Video 5: Regularized Logistic Regression

- **Overview**:
  - Logistic regression can overfit when trained with many features or high-order polynomial features.
  - Regularization helps control overfitting by penalizing large parameter values (\(w_1, w_2, ..., w_n\)).

- **Cost Function with Regularization**:
  - Logistic regression cost function:
    - \( J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \big[ y^{(i)} \log(f(x^{(i)})) + (1 - y^{(i)}) \log(1 - f(x^{(i)})) \big] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2 \)
  - Regularization term:
    - \( \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2 \), where:
      - \( \lambda \): Regularization parameter.
      - Penalizes large values of \(w_j\), encouraging simpler decision boundaries.

- **Effect of Regularization**:
  - Produces decision boundaries that better generalize to unseen examples.
  - Prevents overly complex models even with high-order polynomial features.

- **Gradient Descent Updates**:
  - Parameters \(w_j\) and \(b\) are updated as follows:
    - \( w_j := w_j - \alpha \left( \frac{\partial J}{\partial w_j} \right) \), where:
      - \( \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \big( f(x^{(i)}) - y^{(i)} \big) x_j^{(i)} + \frac{\lambda}{m} w_j \)
    - \( b := b - \alpha \left( \frac{\partial J}{\partial b} \right) \), where:
      - \( \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \big( f(x^{(i)}) - y^{(i)} \big) \)
  - Regularization affects \(w_j\) but not \(b\).

- **Implementation Notes**:
  - Gradient descent with regularization for logistic regression is similar to regularized linear regression.
  - The main difference is the use of the sigmoid function for logistic regression:
    - \( f(x) = \sigma(z) = \frac{1}{1 + e^{-z}} \), where \( z = w \cdot x + b \).


- **Key Takeaways**:
  - Regularized logistic regression helps prevent overfitting while maintaining generalization.
  - Balancing \( \lambda \) is crucial to achieve an optimal trade-off between fitting and regularization.

- **Conclusion**:
  - Congratulations on completing Week 3 and the first course of this specialization!
  - Skills in linear regression, logistic regression, and handling overfitting are highly valuable for real-world machine learning applications.
  - In the next course, you will explore neural networks and deep learning, building on the foundational concepts learned here.
