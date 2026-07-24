"""
generate_data.py
-----------------
Generates a realistic synthetic retail customer dataset for segmentation.
Mimics the structure of common customer-analytics data (e.g. mall/e-commerce
customer records): demographic + behavioral fields with natural cluster
structure baked in (so the pipeline has something meaningful to find).
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_CUSTOMERS = 500


def generate_customers(n=N_CUSTOMERS):
    segments = []

    # Segment 1: Young, low income, high spending (impulsive spenders)
    segments.append(pd.DataFrame({
        "Age": RNG.normal(24, 4, n // 5).clip(18, 35),
        "Annual_Income_k": RNG.normal(28, 6, n // 5).clip(15, 45),
        "Spending_Score": RNG.normal(78, 10, n // 5).clip(50, 100),
        "Annual_Purchases": RNG.normal(45, 8, n // 5).clip(20, 70),
    }))

    # Segment 2: Middle-aged, high income, high spending (premium customers)
    segments.append(pd.DataFrame({
        "Age": RNG.normal(42, 6, n // 5).clip(30, 60),
        "Annual_Income_k": RNG.normal(95, 12, n // 5).clip(70, 140),
        "Spending_Score": RNG.normal(82, 8, n // 5).clip(55, 100),
        "Annual_Purchases": RNG.normal(60, 10, n // 5).clip(30, 90),
    }))

    # Segment 3: Older, moderate income, low spending (conservative savers)
    segments.append(pd.DataFrame({
        "Age": RNG.normal(58, 7, n // 5).clip(45, 75),
        "Annual_Income_k": RNG.normal(55, 10, n // 5).clip(35, 80),
        "Spending_Score": RNG.normal(22, 8, n // 5).clip(1, 40),
        "Annual_Purchases": RNG.normal(15, 5, n // 5).clip(2, 30),
    }))

    # Segment 4: Young, high income, low spending (frugal professionals)
    segments.append(pd.DataFrame({
        "Age": RNG.normal(30, 5, n // 5).clip(22, 40),
        "Annual_Income_k": RNG.normal(90, 10, n // 5).clip(65, 120),
        "Spending_Score": RNG.normal(25, 9, n // 5).clip(1, 45),
        "Annual_Purchases": RNG.normal(20, 6, n // 5).clip(5, 35),
    }))

    # Segment 5: Middle-aged, low income, moderate spending (budget-conscious regulars)
    remaining = n - 4 * (n // 5)
    segments.append(pd.DataFrame({
        "Age": RNG.normal(38, 8, remaining).clip(25, 55),
        "Annual_Income_k": RNG.normal(35, 7, remaining).clip(20, 50),
        "Spending_Score": RNG.normal(50, 10, remaining).clip(30, 70),
        "Annual_Purchases": RNG.normal(35, 7, remaining).clip(15, 55),
    }))

    df = pd.concat(segments, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df["CustomerID"] = [f"CUST{1000+i}" for i in range(len(df))]
    df["Gender"] = RNG.choice(["Male", "Female"], size=len(df))

    df["Age"] = df["Age"].round(0).astype(int)
    df["Annual_Income_k"] = df["Annual_Income_k"].round(1)
    df["Spending_Score"] = df["Spending_Score"].round(1)
    df["Annual_Purchases"] = df["Annual_Purchases"].round(0).astype(int)

    cols = ["CustomerID", "Gender", "Age", "Annual_Income_k",
            "Spending_Score", "Annual_Purchases"]
    return df[cols]


if __name__ == "__main__":
    df = generate_customers()
    df.to_csv("data/customer_data.csv", index=False)
    print(f"Generated {len(df)} customer records -> data/customer_data.csv")
    print(df.head())
