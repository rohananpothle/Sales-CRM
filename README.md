# 📊 CRM Data Analytics Project (Python)

## 👤 Role Assumed

**Data Analyst with 4+ years of real-time industry experience** training **10 interns** using real CRM datasets.

---

## 🎯 Project Objective

Perform an **end-to-end data analytics workflow** using **only Python** on CRM datasets to:

* Understand business problems
* Clean and transform data
* Perform EDA
* Derive KPIs
* Generate insights
* (Optional) Build predictive models

---

## 🗂️ Datasets Used

* `Leads.csv`
* `Contacts.csv`
* `Accounts.csv`
* `Deals.csv`
* `Activities.csv`

These represent a **real-world CRM system** used by Sales & Marketing teams.

---

## 🛠️ Tech Stack

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn (optional)
* Jupyter Notebook

---

## 🧠 Business Questions

* Which lead sources convert best?
* Which activities drive deal closure?
* Which accounts generate maximum revenue?
* What factors influence deal success?

---

## 🔁 End-to-End Analytics Workflow

### 1️⃣ Business Understanding

Understand CRM entities, stakeholders, and KPIs before coding.

---

### 2️⃣ Data Loading

```python
import pandas as pd

leads = pd.read_csv('Leads.csv')
contacts = pd.read_csv('Contacts.csv')
accounts = pd.read_csv('Accounts.csv')
deals = pd.read_csv('Deals.csv')
activities = pd.read_csv('Activities.csv')
```

---

### 3️⃣ Initial Data Exploration

```python
leads.head()
leads.info()
leads.describe()
```

---

### 4️⃣ Data Quality Checks

```python
leads.isna().sum()
leads.duplicated().sum()
```

---

### 5️⃣ Data Cleaning

* Handle missing values
* Fix data types
* Remove duplicates

```python
leads['Budget'].fillna(leads['Budget'].median(), inplace=True)
leads.drop_duplicates(inplace=True)
```

---

### 6️⃣ Feature Engineering

```python
leads['Created_Date'] = pd.to_datetime(leads['Created_Date'])
leads['Lead_Age_Days'] = (pd.Timestamp.today() - leads['Created_Date']).dt.days
```

---

### 7️⃣ Data Integration (Joins)

```python
crm_data = pd.merge(leads, deals, on='Lead_ID', how='left')
```

---

### 8️⃣ Exploratory Data Analysis (EDA)

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=leads, x='Lead_Status')
plt.show()
```

---

### 9️⃣ KPI Calculations

```python
conversion_rate = deals['Deal_Status'].value_counts(normalize=True) * 100
revenue_by_account = deals.groupby('Account_ID')['Deal_Amount'].sum()
```

---

### 🔟 Insights & Findings

* High-quality leads ≠ high-volume leads
* Follow-up activities strongly impact closure rate
* Few accounts contribute majority of revenue

---

### 1️⃣1️⃣ Optional: Predictive Analytics

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = crm_data[['Budget', 'Lead_Age_Days']]
y = crm_data['Deal_Closed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## 📦 Project Deliverables

* Cleaned datasets
* Jupyter Notebook (`CRM_Analysis.ipynb`)
* KPI summary
* Visual insights
* Business recommendations

---

## 👥 Intern Task Distribution

| Interns | Responsibility        |
| ------- | --------------------- |
| 1–2     | Data Cleaning         |
| 3–4     | EDA                   |
| 5–6     | KPI Calculations      |
| 7–8     | Visualization         |
| 9       | Documentation         |
| 10      | Business Presentation |

---

## 🧾 Repository Structure

```
CRM-Data-Analytics/
│── data/
│   ├── Leads.csv
│   ├── Contacts.csv
│   ├── Accounts.csv
│   ├── Deals.csv
│   └── Activities.csv
│
│── notebooks/
│   └── CRM_Analysis.ipynb
│
│── README.md
```

---

## ⭐ Key Learning for Interns

> "Python is just a tool. A Data Analyst’s real value lies in business understanding, data thinking, and storytelling."

---

✅ **This README is GitHub-ready.**
