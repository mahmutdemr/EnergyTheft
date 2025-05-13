import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Customer types and behavior function
customer_types = {
    "normal": {"desc": "Normal consumption", "label": 0},
    "night_thief": {"desc": "Night thief", "label": 1},
    "peak_thief": {"desc": "Evening thief", "label": 1},
    "sudden_spike": {"desc": "Sudden fluctuations", "label": 0},
    "noisy": {"desc": "Noisy signal", "label": 0},
    # "gradual_thief": {"desc": "Gradually increasing theft", "label": 1},
    # "specific_thief": {"desc": "Theft on specific days", "label": 1},
    # "variable_thief": {"desc": "Variable time theft", "label": 1}
}

def apply_customer_behavior(customer_type, measured_kw, stolen_kw, time_index):
    if customer_type == "normal":
        for i, ts in enumerate(time_index):
            if ts.weekday() >= 5:  # Saturday-Sunday
                measured_kw[i] *= np.random.uniform(1.1, 1.3)
            else:
                measured_kw[i] *= np.random.uniform(0.9, 1.1)

    elif customer_type == "night_thief":
        for day in pd.date_range(start=time_index[0], end=time_index[-1], freq="D"):
            if np.random.rand() < 0.7:  # 70% chance of theft
                start_hour = np.random.randint(22, 2 + 24) % 24
                end_hour = (start_hour + np.random.randint(2, 4)) % 24
                indices = [
                    i for i, ts in enumerate(time_index)
                    if ts.date() == day.date() and
                    (start_hour <= ts.hour < end_hour if start_hour < end_hour else ts.hour >= start_hour or ts.hour < end_hour)
                ]
                for idx in indices:
                    stolen_kw[idx] = np.random.uniform(5, 15)
                    measured_kw[idx] *= np.random.uniform(0.6, 0.75)

    elif customer_type == "peak_thief":
        for day in pd.date_range(start=time_index[0], end=time_index[-1], freq="D"):
            if np.random.rand() < 0.6:
                if np.random.rand() < 0.5:
                    theft_hours = range(18, 23)
                else:
                    start = np.random.randint(0, 20)
                    theft_hours = range(start, start + 3)
                indices = [
                    i for i, ts in enumerate(time_index)
                    if ts.date() == day.date() and ts.hour in theft_hours
                ]
                for idx in indices:
                    stolen_kw[idx] = np.random.uniform(8, 18)
                    measured_kw[idx] *= np.random.uniform(0.55, 0.75)

    elif customer_type == "sudden_spike":
        events = np.random.randint(5, 15)
        for _ in range(events):
            idx = np.random.randint(0, len(measured_kw) - 4)
            measured_kw[idx:idx + 2] *= np.random.uniform(1.5, 2.5)
            measured_kw[idx + 2:idx + 4] *= np.random.uniform(0.3, 0.6)

    elif customer_type == "noisy":
        drift = np.linspace(1, np.random.uniform(0.95, 1.05), len(measured_kw))
        noise = np.random.normal(0, 0.1, len(measured_kw))
        measured_kw *= (drift + noise)

    return measured_kw, stolen_kw

"""
    elif customer_type == "gradual_thief":
        theft_start_day = np.random.randint(3, 6)  # Start after 3rd day
        for day_index, day in enumerate(pd.date_range(start=time_index[0], end=time_index[-1], freq="D")):
            if day_index >= theft_start_day:
                scale = 1 + (day_index - theft_start_day) * 0.1  # Increasing theft amount
                for i, ts in enumerate(time_index):
                    if ts.date() == day.date() and 18 <= ts.hour <= 22:
                        stolen_kw[i] = np.random.uniform(4, 8) * scale
                        measured_kw[i] *= np.random.uniform(0.6, 0.75)

    elif customer_type == "specific_thief":
        pattern = np.random.choice(["week_start", "weekend"])
        for i, ts in enumerate(time_index):
            if (pattern == "week_start" and ts.weekday() == 0) or (pattern == "weekend" and ts.weekday() >= 5):
                if 18 <= ts.hour <= 22:
                    stolen_kw[i] = np.random.uniform(6, 12)
                    measured_kw[i] *= np.random.uniform(0.5, 0.75)

    elif customer_type == "variable_thief":
        for day in pd.date_range(start=time_index[0], end=time_index[-1], freq="D"):
            if np.random.rand() < 0.6:
                start_hour = np.random.randint(0, 24)
                duration = np.random.randint(1, 4)
                end_hour = (start_hour + duration) % 24
                indices = [
                    i for i, ts in enumerate(time_index)
                    if ts.date() == day.date() and
                    (start_hour <= ts.hour < end_hour if start_hour < end_hour else ts.hour >= start_hour or ts.hour < end_hour)
                ]
                for idx in indices:
                    stolen_kw[idx] = np.random.uniform(5, 10)
                    measured_kw[idx] *= np.random.uniform(0.6, 0.8)

    return measured_kw, stolen_kw
"""

def generate_random_customers(num_customers, time_steps=96*2, seed=None, theft_ratio=0.0):
    if seed is not None:
        np.random.seed(seed)

    data = []
    time_index = pd.date_range(start="2025-01-01", periods=time_steps, freq="15min")

    # Thief and non-thief customer types
    theft_types = ["night_thief", "peak_thief"]
    non_theft_types = ["normal", "sudden_spike", "noisy"]

    num_theft_customers = int(num_customers * theft_ratio)
    theft_customer_indices = np.random.choice(num_customers, size=num_theft_customers, replace=False)

    # Base load profile
    base_profile = []
    for ts in time_index:
        hour = ts.hour
        if 0 <= hour < 6:
            base_profile.append(0.5)
        elif 6 <= hour < 12:
            base_profile.append(0.7)
        elif 12 <= hour < 18:
            base_profile.append(0.9)
        elif 18 <= hour < 23:
            base_profile.append(1.2)
        else:
            base_profile.append(0.6)
    base_profile = np.array(base_profile)

    for i in range(num_customers):
        customer_id = f"c{i+1}"

        # Customers at specified indices will be thieves
        if i in theft_customer_indices:
            customer_type = np.random.choice(theft_types)
            label = 1
        else:
            customer_type = np.random.choice(non_theft_types)
            label = 0

        base_kw = np.random.randint(80, 121)
        measured_kw = base_kw * base_profile * np.random.normal(1.0, 0.05, time_steps)
        measured_kw = np.clip(measured_kw, 0, None)
        stolen_kw = np.zeros(time_steps)

        measured_kw, stolen_kw = apply_customer_behavior(customer_type, measured_kw, stolen_kw, time_index)

        customer_df = pd.DataFrame({
            "Customer ID": customer_id,
            "Customer Type": customer_type,
            "Label": label,
            "Timestamp": time_index,
            "Measured kW": np.round(measured_kw, 3),
            "Stolen kW": np.round(stolen_kw, 3)
        })

        data.append(customer_df)

    return pd.concat(data, ignore_index=True)

# Generate training and test data
train_data = generate_random_customers(num_customers=10, seed=40, theft_ratio=0.4)
train_data["True kW"] = train_data["Measured kW"] + train_data["Stolen kW"]

transformer = train_data.groupby("Timestamp")["True kW"].sum().reset_index()
train_data.to_csv("Training.csv", index=False)
transformer.to_csv("Transformer.csv", index=False)

test_data = generate_random_customers(num_customers=10, seed=21, theft_ratio=0.4)
test_data["True kW"] = test_data["Measured kW"] + test_data["Stolen kW"]
transformer_test = test_data.groupby("Timestamp")["True kW"].sum().reset_index()
test_data.to_csv("Test.csv", index=False)
transformer_test.to_csv("TransformerTest.csv", index=False)

# Plot transformer power profile
customer_id = "c8"  # Change customer_id to plot another customer's consumption
example = train_data[train_data["Customer ID"] == customer_id]

plt.figure(figsize=(15, 5))
plt.plot(transformer["Timestamp"], transformer["True kW"], label="Transformer", color='black')
plt.legend()
plt.title("Transformer Power Profile")
plt.xlabel("Time")
plt.ylabel("kW")
plt.grid()
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def create_features(data):
    features = []

    for customer_id, group in data.groupby("Customer ID"):
        measured = group["Measured kW"]
        timestamps = pd.to_datetime(group["Timestamp"])
        label = group["Label"].iloc[0]

        df_temp = pd.DataFrame({
            "measured": measured.values,
            "hour": timestamps.dt.hour,
            "weekday": timestamps.dt.weekday,
            "week": timestamps.dt.isocalendar().week,
            "year": timestamps.dt.year
        })

        # Spike threshold
        mean_kw = measured.mean()
        std_kw = measured.std()
        spike_threshold = mean_kw + 2 * std_kw
        spikes = df_temp[df_temp["measured"] > spike_threshold]

        # Weekly grouping
        weekly = df_temp.groupby(["year", "week"])["measured"]
        weekly_medians = weekly.median()
        weekly_stds = weekly.std()

        # Lag feature: previous week's average
        weekly_means = weekly.mean()
        lag_week_mean = weekly_means.shift(1).dropna().mean()

        # Seasonal usage ratio (weekend / total)
        weekend_total = df_temp[df_temp["weekday"] >= 5]["measured"].sum()
        total_kwh = df_temp["measured"].sum()
        seasonal_ratio = weekend_total / total_kwh if total_kwh > 0 else 0

        feature_dict = {
            "Customer ID": customer_id,
            "mean_kw": mean_kw,
            "max_kw": measured.max(),
            "min_kw": measured.min(),
            "std_kw": std_kw,
            "median_kw": measured.median(),
            "skew_kw": measured.skew(),
            "kurtosis_kw": measured.kurt(),
            "evening_avg_kw": df_temp[df_temp["hour"].between(18, 22)]["measured"].mean(),
            "weekend_avg_kw": df_temp[df_temp["weekday"] >= 5]["measured"].mean(),
            "weekday_avg_kw": df_temp[df_temp["weekday"] < 5]["measured"].mean(),
            "peak_valley_diff": measured.max() - measured.min(),
            "Label": label,
            "weekly_median_avg": weekly_medians.mean(),
            "weekly_std_avg": weekly_stds.mean(),
            "spike_count": len(spikes),
            "avg_spike_kw": spikes["measured"].mean() if not spikes.empty else 0,
            "lag_week_mean": lag_week_mean if not np.isnan(lag_week_mean) else 0,
            "seasonal_weekend_ratio": seasonal_ratio
        }

        features.append(feature_dict)

    return pd.DataFrame(features)

# Create features for training and test data
train_features = create_features(train_data)
test_features = create_features(test_data)

X_train = train_features.drop(columns=["Customer ID", "Label"])
y_train = train_features["Label"]

X_test = test_features.drop(columns=["Customer ID", "Label"])
y_test = test_features["Label"]  # For accuracy evaluation if needed

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Save predictions
prediction_df = pd.DataFrame({
    "Customer ID": test_features["Customer ID"],
    "Predicted Label": y_pred,
    "True Label": y_test
})

print(prediction_df)
prediction_df.to_csv("predictions.csv", index=False)