# Customer Segmentation using Scikit-Learn K-Means

An end-to-end Python data science project that groups customers into distinct actionable business segments using machine learning (`scikit-learn`). This project bypasses heavy BI tools like Power BI by extracting insights and generating visualizations entirely within Python.

## 📋 Project Overview
This project processes transaction and customer profile data (`customer_segmentation_data.csv`) to cluster a user base of 1,000 customers. By applying feature scaling and the K-Means clustering algorithm, the pipeline uncovers natural consumer groupings based on financial behavior.

### Key Features
* **Data Preprocessing:** Standardizes varying scales (Annual Income vs. Spending Score) using `StandardScaler`.
* **Machine Learning Clustering:** Leverages `KMeans` to dynamically identify mathematical patterns.
* **Business Mapping:** Automatically translates numeric cluster labels into strategic business definitions (e.g., *VIPs*, *Savers*).
* **Pure Python Reporting:** Outputs real-time macro KPI summary statistics directly to the terminal alongside comprehensive visual scatter plots.

---

## 🚀 Identified Customer Segments

The K-Means algorithm divides the target market into four distinct behavioral categories:

1. **High Income, High Spenders (VIPs):** Premium consumers with large capital reserves and a strong affinity for high transaction frequencies. Target with loyalty rewards and exclusive early access.
2. **High Income, Low Spenders (Savers):** High earners who spend conservatively. Target with value-driven bundles, utility marketing, and premium functionality arguments.
3. **Low Income, High Spenders:** Younger or impulse-driven demographics with smaller budgets but highly active spending profiles. Target with flash sales, discount triggers, and flexible financing.
4. **Low Income, Low Spenders:** Value-conscious budget shoppers. Target with essential baselines and high-necessity product categories.

---

## 🛠️ Built With

* **Python 3** - Programming Language
* **Pandas** - Data manipulation and profiling
* **Scikit-Learn** - Machine learning pipeline (Scaling & K-Means)
* **Matplotlib & Seaborn** - Static dashboard and visualization rendering

---

## ⚙️ Installation & Setup

Follow these steps to run the script locally in your **VS Code** development environment:

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
