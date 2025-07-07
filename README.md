## 🚗 Dynamic Pricing Models: Linear vs XGBoost

 This repo has **two pricing models** for predicting ride or parking prices:

* ✅ **Model 1:** A simple **Linear Formula**
* 🚀 **Model 2:** A smarter **XGBoost ML Model**

---

## 📂 What’s Inside

| Part                 | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| `dataset.csv`        | Your input data: occupancy, capacity, queue length, traffic, etc. |
| `linear_pricing.py`  | Basic model with a simple linear formula.                         |
| `xgboost_pricing.py` | Advanced model using XGBoost regression.                          |

---

## ⚡️ 1️⃣ Linear Pricing Formula

### 🧐 What it does

A hand-crafted formula:

> **Price = Base Price + Occupancy Effect + Queue Effect**

Where:

* `Base Price` = \$10.00
* `Occupancy Effect` = more cars → higher price
* `Queue Effect` = longer wait → higher price

It uses:

```python
BASE_PRICE = 10.0
alpha = 8.0  # occupancy weight
queue_weight = 0.5  # queue weight
```

### 🛠️ How it works

1. Calculates `occupancy_ratio = Occupancy / Capacity`
2. Computes `target_price` with the formula above
3. Predicts price with the **same** formula → so your MAE will be zero (duh!)

### 📊 Visuals

* **Target vs Predicted** — sanity check (should be a perfect diagonal line).
* **Price vs Occupancy Ratio** — shows how price goes up with occupancy.

---

## 🤖 2️⃣ XGBoost Dynamic Pricing Model

### 🧐 What it does

Uses machine learning (XGBoost regressor) to learn how **real-world factors** affect price. It’s more flexible because it considers:

* Occupancy ratio
* Queue length
* Traffic conditions (low/average/high)
* Is it a special day?
* Vehicle type (car/bike/truck)
* Time of day
* Day of week

### 🧩 How it works

1. **Feature Engineering:**

   * Extract `Hour` and `DayOfWeek`
   * Encode traffic & vehicle type
   * Calculate a **Demand Factor** (custom function) → combines all factors using math & sigmoid magic.

2. **Target Price:**

   ```python
   target_price = BASE_PRICE * demand_factor
   ```

3. **Train-Test Split:**

   * Train XGBoost on 80% of data
   * Test on 20% → get RMSE

4. **Prediction:**

   * Clips final prices within `0.6x` to `2.0x` base price.

5. **Real-Time Function:**

   * Plug in live values → get dynamic price.

---

## 🔍 Visuals

* **Actual vs Predicted:** scatter plot to see how well XGBoost fits.
* **Feature Importance:** bar chart to see which features matter most.
* **Hourly Pricing Pattern:** see how price changes by hour.
* **Price Distribution:** see price spread & base price limits.

---

## ✅ Key Takeaways

| Linear Formula        | XGBoost Model                    |
| --------------------- | -------------------------------- |
| Easy to understand    | Learns patterns automatically    |
| Deterministic         | Adapts to real-world variation   |
| Good for sanity check | Better accuracy for complex data |

---

## 💡 How to Run

1. Make sure you have:

   ```bash
   pip install pandas numpy matplotlib xgboost scikit-learn
   ```

2. Put `dataset.csv` in the same folder.

3. Run:

   ```bash
   python linear_pricing.py
   python xgboost_pricing.py
   ```

4. Check your console & the plots!

---

## 🚀 Future Ideas

* Add weather data 🌦️
* Real-time API to serve prices ⚡️
* Experiment with other models (LightGBM, Random Forests, Neural Nets)


