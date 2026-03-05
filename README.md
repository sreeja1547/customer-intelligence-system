# Customer Intelligence System

### Customer Churn Prediction & Customer Segmentation

## Project Overview

Customer churn is a major challenge for subscription-based businesses such as telecom companies.
This project builds an end-to-end **Customer Intelligence System** that:

* Predicts **which customers are likely to churn** using supervised learning
* Segments customers into **meaningful behavioral groups** using unsupervised learning
* Combines both results to generate **actionable business retention strategies**

The goal is not only to build machine learning models, but also to support **data-driven business decision making**.

---

## Business Objectives

* Identify customers with **high churn risk**
* Understand **different customer behavior patterns**
* Identify **high-value customer segments**
* Design **segment-specific retention strategies**

---

## Dataset

* **Dataset:** Telco Customer Churn Dataset
* **Domain:** Telecommunications
* **Records:** 7,000+ customers

**Key Features**

* Customer demographics
* Service usage details
* Contract and billing information
* Payment methods
* Churn label (Yes / No)

---

## Technologies & Tools

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* FastAPI
* Joblib

---

## Project Approach

### Data Preprocessing & Exploratory Data Analysis

* Handled missing values and corrected data types
* Performed exploratory data analysis to understand customer behavior
* Analyzed relationships between tenure, charges, and churn

---

### Churn Prediction (Supervised Learning)

* Built a preprocessing pipeline including:

  * Numerical feature scaling
  * Categorical feature encoding
* Trained a **Logistic Regression model** to predict churn

**Outcome:**
Identified customers with a high probability of churn.

---

### Customer Segmentation (Unsupervised Learning)

* Removed churn label for unbiased clustering
* Applied **K-Means clustering** on behavioral features
* Segmented customers into **five clusters**

**Outcome:**
Each customer was assigned to a behavioral segment.

---

### Combined Business Intelligence Analysis

* Calculated **churn rate per segment**
* Calculated **average revenue per segment**
* Identified:

  * High-risk segments
  * High-value segments
  * Stable customer groups

---

## Key Insights

| Segment   | Churn Rate | Revenue Level | Interpretation             |
| --------- | ---------- | ------------- | -------------------------- |
| Cluster 1 | ~47%       | Low           | High churn risk customers  |
| Cluster 4 | ~50%       | Medium        | Highest churn risk         |
| Cluster 2 | ~16%       | Very High     | Loyal high-value customers |
| Cluster 0 | ~7%        | Medium        | Stable long-term customers |
| Cluster 3 | ~7%        | Low           | Low-cost stable customers  |

---

## Business Recommendations

**High Churn Segments (Clusters 1 & 4)**

* Targeted retention campaigns
* Promotional offers
* Contract upgrades

**High-Value Segment (Cluster 2)**

* Loyalty programs
* Premium support services

**Stable Segments (Clusters 0 & 3)**

* Maintain service quality
* Cross-selling and upselling opportunities

---

## Business Impact

This system helps organizations:

* Reduce customer churn
* Protect high-value customers
* Improve retention strategies
* Increase overall customer lifetime value

---

## Conclusion

By combining **supervised learning**, **unsupervised learning**, and **business analysis**, this project demonstrates how machine learning can be applied to solve real-world customer retention problems.

---

## Running the API

Start the FastAPI server:

```
uvicorn api.app:app --reload
```

Open the API documentation:

```
http://127.0.0.1:8000/docs
```

---

## Project Structure

```
customer-intelligence-system
│
├── api
│   └── app.py
│
├── data
│   └── telco_dataset.csv
│
├── model
│   └── customer_model.pkl
│
├── src
│   └── customer_intelligence_system.py
│
├── requirements.txt
└── README.md
