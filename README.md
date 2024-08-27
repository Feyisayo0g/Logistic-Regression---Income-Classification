# Logistic-Regression---Income-Classification
Income Classification Using Logistic Regression

### Background
In this project, we aimed to build a predictive model using logistic regression to classify individuals based on whether their income exceeds $50,000, using census data from the 1994 UCI Machine Learning Repository. The dataset includes various demographic and financial attributes, making it an ideal candidate for exploring classification techniques and gaining insights into socioeconomic factors influencing income levels.

### Executive Summary
The project involved analyzing and modeling a dataset containing demographic and income-related features to predict whether an individual earns more than $50,000 annually. By applying logistic regression, we identified key predictors and evaluated the model's performance. The analysis revealed significant factors contributing to income levels and provided a clear understanding of how demographic variables influence income distribution.

### Goals of Analysis
- To predict whether an individual's income exceeds $50,000 using logistic regression.
- To identify and understand the most significant predictors of income levels.
- To evaluate the model's performance and assess its ability to generalize to unseen data.

### Methodology
- Exploratory Data Analysis (EDA): Conducted EDA to understand the distribution of income and the relationships between predictor variables.
- Data Preparation: Transformed categorical variables into dummy variables and scaled continuous features to meet the assumptions of logistic regression.
- Model Development: Built a logistic regression model using the LogisticRegression function from scikit-learn, with L1 regularization to handle multicollinearity.
- Model Evaluation: Assessed the model's performance using accuracy score, confusion matrix, and ROC AUC score. Visualized model coefficients to interpret the importance of each feature.

### Key Insights
- Significant Predictors: Variables such as education, age, capital-gain, and hours-per-week were significant predictors of income, with higher education and capital gains strongly associated with earning more than $50,000.

- Model Performance: The logistic regression model achieved an accuracy of approximately 83.67% on the test set, with a ROC AUC score of 0.90, indicating a robust model with good discrimination capability.

- Feature Importance: The analysis revealed that capital gains and education level are the most influential factors, with higher coefficients indicating a greater likelihood of higher income.

### Recommendation
To further improve the model, consider exploring additional techniques such as feature engineering, interaction terms, or alternative classification algorithms like Random Forest or Gradient Boosting. Additionally, addressing class imbalance through techniques like SMOTE (Synthetic Minority Over-sampling Technique) could enhance the model's ability to correctly classify minority classes. The insights from this model can be leveraged in policy-making, targeted marketing, or socioeconomic studies to understand income disparities.

