# Week 2: Multiple Linear Regression

## Video 1: Multiple Features

- **Expanding Linear Regression**:
  - Traditional linear regression uses a single feature (\(x\)) to predict an outcome (\(y\)).
  - Multiple linear regression includes multiple features (\(x_1, x_2, ..., x_n\)) to improve predictions.

- **Notation**:
  - Features (\(x_j\)) represent the inputs (e.g., house size, number of bedrooms, etc.).
  - Number of features = \(n\).
  - Example:
    - For a house with size 1416 sq. ft., 3 bedrooms, 2 floors, and age 40:
      - Feature vector: \(X^{(2)} = [1416, 3, 2, 40]\).
  - Model parameters (\(w_1, w_2, ..., w_n, b\)) represent the weights and bias.

- **Model Representation**:
  - \(f_{w,b}(X) = w_1x_1 + w_2x_2 + ... + w_nx_n + b\).
  - Example coefficients:
    - \(w_1 = 0.1\): House size increases price by $100 per sq. ft.
    - \(w_2 = 4\): Each bedroom adds $4,000.
    - \(w_3 = 10\): Each floor adds $10,000.
    - \(w_4 = -2\): Each year of age decreases price by $2,000.
  - Base price (\(b\)) = $80,000.

- **Vector Notation**:
  - Parameters vector: \(\mathbf{w} = [w_1, w_2, ..., w_n]\).
  - Features vector: \(\mathbf{x} = [x_1, x_2, ..., x_n]\).
  - Compact model representation:
    - \(f_{w,b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b\).

- **Key Terminology**:
  - **Dot Product**: Summation of corresponding elements of two vectors.
  - **Multiple Linear Regression**: Uses multiple features; distinct from "multivariate regression."

---

## Video 2: Vectorization Part 1

- **Introduction to Vectorization**:
  - A programming technique that improves efficiency and readability by leveraging mathematical operations on vectors.
  - Enables the use of libraries (e.g., NumPy) and hardware (e.g., GPUs) for faster computations.

- **Examples**:
  - Without vectorization:
    - Use loops to compute dot products, inefficient for large \(n\).
  - With vectorization:
    - NumPy function `np.dot(w, x)` performs dot products efficiently.
    - Single-line implementation: \(f_p = \text{np.dot}(w, x) + b\).

- **Benefits**:
  - Shorter, cleaner code.
  - Faster execution due to parallel processing in hardware.


  - Especially useful for large datasets or models with many features.

---

## Video 3: Vectorization Part 2

- **How Vectorization Works**:
  - Without vectorization:
    - Operations are sequential (e.g., one feature at a time in a loop).
  - With vectorization:
    - Parallel processing computes all feature operations simultaneously.
    - Aggregates results efficiently.

- **Example**:
  - Updating parameters in gradient descent:
    - Without vectorization:
      - Use a loop: \(w_j = w_j - \alpha \cdot d_j\) (for all \(j\)).
    - With vectorization:
      - Single-line: \(\mathbf{w} = \mathbf{w} - \alpha \cdot \mathbf{d}\).

- **Efficiency**:
  - Significant time savings for large \(n\) or datasets.
  - Difference between code running in minutes vs. hours.

- **Hands-On Lab**:
  - Practice using NumPy for vectorized operations.
  - Measure runtime differences between vectorized and unvectorized code.
  - Learn to create and manipulate NumPy arrays.


## Video 4: Gradient Descent for Multiple Linear Regression

- **Overview**:
  - Combines gradient descent, multiple linear regression, and vectorization to efficiently update parameters (\(w\) and \(b\)).

- **Model Representation**:
  - Parameters:
    - \(w\): A vector containing \(w_1, w_2, ..., w_n\).
    - \(b\): A single bias parameter.
  - Model equation:
    - \(f_{w,b}(x) = \mathbf{w} \cdot \mathbf{x} + b\).

- **Cost Function**:
  - Defined as \(J(w, b)\), a function of the parameter vector \(\mathbf{w}\) and scalar \(b\).
  - Gradient descent minimizes \(J\) iteratively.

- **Gradient Descent Update Rules**:
  - For a single feature (univariate regression):
    - \(w := w - \alpha \cdot \frac{\partial J}{\partial w}\)
    - \(b := b - \alpha \cdot \frac{\partial J}{\partial b}\)
  - For multiple features (\(n \geq 2\)):
    - \(w_j := w_j - \alpha \cdot \frac{\partial J}{\partial w_j}\), for \(j = 1, 2, ..., n\).
    - \(b := b - \alpha \cdot \frac{\partial J}{\partial b}\).

- **Key Insights**:
  - The error term (\(f(x) - y\)) is similar across univariate and multivariate cases.
  - Updates for all \(w_j\) are computed for \(j = 1\) to \(n\).
  - Gradient descent scales efficiently for multiple features.

- **Alternative: Normal Equation**:
  - Solves for \(w\) and \(b\) using a direct formula without iterations.
  - Limitations:
    - Not applicable to other algorithms (e.g., logistic regression, neural networks).
    - Computationally expensive for large \(n\).
    - Rarely used in practice but implemented in some machine learning libraries.

- **Practical Notes**:
  - Gradient descent is more versatile and widely applicable than the normal equation.
  - Modern libraries may use the normal equation for backend computations.


---

## Summary

- Multiple Linear Regression incorporates multiple features for better predictions.
- Vectorization is a critical technique for writing efficient, scalable machine learning code.
- Tools like NumPy simplify implementation and enable faster computation through parallel processing.


# Week 2: Gradient Descent in Practice

## Video 1: Feature Scaling Part 1

- **Concept**:
  - Feature scaling accelerates gradient descent by normalizing features to comparable ranges.

- **Example**:
  - \(x_1\) (size in square feet): 300 to 2000.
  - \(x_2\) (number of bedrooms): 0 to 5.
  - Features with larger ranges (\(x_1\)) lead to smaller parameters (\(w_1\)).
  - Features with smaller ranges (\(x_2\)) lead to larger parameters (\(w_2\)).

- **Challenges in Gradient Descent**:
  - Features with different scales lead to contour plots with elongated ellipses.
  - Gradient descent bounces between contours, slowing convergence.

- **Solution**:
  - Rescale features so all features range from 0 to 1.
  - Contours become circular, enabling a direct path to the minimum.

---

## Video 2: Feature Scaling Part 2

- **Implementation**:
  - **Rescaling**:
    - Divide each feature by its maximum value.
    - Example: \(x_1 / 2000\), \(x_2 / 5\).
  - **Mean Normalization**:
    - Subtract the mean (\(\mu\)) from each feature and divide by the range:
      - \(x' = \frac{x - \mu}{\text{max} - \text{min}}\).
    - Centers features around 0 and scales them to -1 to 1.

- **Alternative: Z-Score Normalization**:
  - Formula: \(x' = \frac{x - \mu}{\sigma}\), where \(\sigma\) is the standard deviation.
  - Scales features to have mean 0 and variance 1.

- **Best Practices**:
  - Aim for feature ranges between -1 and 1 or close to that range.
  - Rescaling large or small features significantly improves gradient descent performance.

---

## Video 3: Checking Gradient Descent for Convergence

- **Monitoring Convergence**:
  - Plot the cost function \(J\) over iterations (learning curve).
  - Gradient descent is working properly if \(J\) decreases consistently.

- **Signs of Convergence**:
  - \(J\) levels off after several iterations.
  - Use a small threshold (\(\epsilon\)) to determine when the cost changes minimally (e.g., \(< 0.001\)).

- **Debugging Tips**:
  - If \(J\) increases, reduce the learning rate (\(\alpha\)) or check for bugs in code.
  - A small \(\alpha\) ensures \(J\) decreases with every iteration.

---

## Video 4: Choosing the Learning Rate

- **Importance of Learning Rate (\(\alpha\))**:
  - Too small: Slow convergence.
  - Too large: May fail to converge or cause oscillations.

- **Diagnosing Issues**:
  - Cost \(J\) increases or fluctuates: Reduce \(\alpha\).
  - Consistent increase in \(J\): Check for incorrect update rule (e.g., missing minus sign).

- **Choosing \(\alpha\)**:
  - Test values in a range (e.g., 0.001, 0.003, 0.01, 0.03).
  - Aim for the largest value of \(\alpha\) that ensures consistent decrease in \(J\).

- **Practical Steps**:
  - Experiment with small iterations for each \(\alpha\).
  - Choose the largest value that decreases \(J\) effectively.

---

## Video 5: Feature Engineering

- **Concept**:
  - Feature engineering transforms or combines features to improve model performance.

- **Example**:
  - Predict house price using \(x_1\) (lot width) and \(x_2\) (lot depth).
  - Create a new feature \(x_3 = x_1 \times x_2\) (lot area), which is more predictive.

- **Benefits**:
  - Helps models better capture relationships in the data.
  - Enables the model to fit non-linear patterns by designing appropriate features.

- **Best Practices**:
  - Use domain knowledge or intuition to engineer features.
  - Define new features to simplify learning and improve accuracy.

## Video 5: Polynomial Regression

- **Introduction**:
  - Extends multiple linear regression with feature engineering to fit non-linear functions (curves) to data.

- **Concept**:
  - Polynomial regression uses powers of features (e.g., \(x^2, x^3\)) to better fit non-linear relationships.
  - Example:
    - Original feature: \(x\) (size in square feet).
    - Polynomial features: \(x^2\) (size squared), \(x^3\) (size cubed).

- **Applications**:
  - Quadratic Model:
    - Adds \(x^2\) to capture a parabolic relationship.
    - Limitations: May predict prices to decrease at higher sizes (not realistic for housing data).
  - Cubic Model:
    - Adds \(x^3\) for a more realistic upward trend for larger houses.
    - Better fit for data with non-linear relationships.

- **Feature Scaling**:
  - Essential for polynomial regression as feature ranges can vary significantly.
    - Example: \(x\) (1–1,000), \(x^2\) (1–1,000,000), \(x^3\) (1–1,000,000,000).
  - Ensures gradient descent converges efficiently.

- **Alternative Features**:
  - Square Root:
    - Model: \(f(x) = w_1x + w_2\sqrt{x} + b\).
    - Useful for capturing non-linear but non-flattening trends.

- **Choosing Features**:
  - Feature engineering allows flexibility in selecting or designing features.
  - Later courses will introduce methods to systematically evaluate and select features.

- **Practical Implementation**:
  - Use polynomial regression to create better models for data with non-linear relationships.
  - Explore optional lab exercises to:
    - Implement polynomial regression with \(x, x^2, x^3\).
    - Use Scikit-learn for a streamlined implementation.
    - A popular machine learning library for practitioners.
    - Allows linear and polynomial regression with a few lines of code.
    - Encourages understanding algorithms beyond using black-box libraries.

- **Conclusion**:
  - Polynomial regression and feature engineering improve model performance.
  - Practice labs provide hands-on experience with these techniques.
  - Prepares for next week’s transition from regression to classification algorithms.