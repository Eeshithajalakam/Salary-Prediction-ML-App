import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load dataset
print("📂 Loading dataset...")
df = pd.read_csv("Salary_data.csv")

# Step 2: Filter rows where YearsExperience >= 1
df = df[df["YearsExperience"] >= 1]
df["YearsExperience"]=df["YearsExperience"].astype(int)
print(f"✅ Dataset loaded with {len(df)} rows")

# Step 3: Split into features (X) and target (y)
X = df[["YearsExperience"]]
y = df["Salary"]

# Step 4: Train-test split
print("🔄 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print(f"   Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Step 5: Train the Linear Regression model
print("🤖 Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Save the trained model
joblib.dump(model, "salary_model.pkl")
print("💾 Model saved as salary_model.pkl")
print(f"Formula: Salary = {model.coef_[0]:.2f} * YearsExperience + {model.intercept_:.2f}")

# Step 7: Show model details
print("\n📊 Model details:")
print(f"   Coefficient (slope): {model.coef_[0]:.2f}")
print(f"   Intercept: {model.intercept_:.2f}")
