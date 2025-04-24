# # Decoding Energy Consumption Predictions: SHAP in LSTM Models

### The Need for Transparency in Energy Forecasting

In the critical domain of energy consumption forecasting, accurate predictions alone aren't sufficient. Energy managers and operators need to understand how models make decisions to trust and effectively act upon these predictions. In this project, we explore SHAP (SHapley Additive exPlanations) for interpreting an LSTM model's predictions of appliance energy usage. Using a comprehensive energy dataset, we'll build a temporal model, dissect its logic with a custom SHAP implementation, and analyze feature importance across time steps.

## 1. The Dataset: Understanding Energy Consumption Patterns

Our dataset includes 28 features from a low-energy house, capturing appliance energy use and environmental conditions at 10-minute intervals.

**Key Features:**

-   `Appliances`: Energy use in Wh (target variable)
    
-   `lights`: Energy use of light fixtures
    
-   `T1-T9`: Temperature measurements from different areas
    
-   `RH_1-RH_9`: Humidity measurements
    
-   `Press_mm_hg`: Pressure in mmHg
    
-   `Visibility`: Visibility in km
    
-   `Windspeed`: Windspeed in m/s
    

**Temporal Characteristics:**

-   19,735 observations over 4.5 months
    
-   10-minute granularity
    
-   24-timestep window used for forecasting (4-hour lookback)
    

## 2. Building the Predictive LSTM Model

We trained a  **3-layer LSTM network**  to predict appliance energy consumption.

**Preprocessing Steps:**

-   Min-Max scaled all features (0-1 range)
    
-   Created sequential windows of 24 timesteps (4 hours)
    
-   80/20 train-test split maintaining temporal order
    

**Model Architecture:**
```python
class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = F.relu(self.fc1(out[:, -1, :]))  # Last timestep
        return self.fc2(out)
```
**Training Results:**

-   Epochs: 10
    
-   Final Training Loss (MSE): 0.0024
    
-   Convergence shown in training curve in the notebook

## 3. Demystifying the Temporal Model with SHAP

### The Challenge of Temporal SHAP

Traditional SHAP methods struggle with:

1.  Temporal dependencies between features
    
2.  Variable importance across time steps
    
3.  High computational complexity for sequential data
    

### Our Custom LSTM-SHAP Implementation

We developed a specialized SHAP explainer that:

-   Respects temporal dependencies when creating masks
    
-   Efficiently samples from background distribution
    
-   Handles variable-length sequences
    

**Key Innovations:**

1.  **Temporal Masking:**  Generates masks that preserve temporal patterns
    
2.  **Adaptive Sampling:**  Focuses samples on important time regions
    
3.  **Regularized Regression:**  Stable SHAP value estimation
    

```python

class LSTMTimeSeriesSHAP:
    def _generate_temporal_masks(self, nsamples):
        masks = []
        # All features present/absent
        masks.append(np.ones((self.timesteps, self.n_features)))
        masks.append(np.zeros((self.timesteps, self.n_features)))
        
        # Temporal-aware sampling
        for _ in range(nsamples-2):
            mask = np.zeros((self.timesteps, self.n_features))
            active_timesteps = np.random.choice(
                self.timesteps, 
                np.random.randint(1, self.timesteps), 
                replace=False
            )
            for t in active_timesteps:
                active_features = np.random.choice(
                    self.n_features, 
                    np.random.randint(1, self.n_features), 
                    replace=False
                )
                mask[t, active_features] = 1
            masks.append(mask)
        return np.array(masks)
  ```
## 4. Model Validation and Feature Analysis

### Feature Importance Across Time

We computed mean absolute SHAP values to identify globally important features:

```python
def global_temporal_importance(explainer, instances):
    shap_values = []
    for instance in instances:
        sv = explainer.shap_values(instance.unsqueeze(0))
        shap_values.append(sv)
    return np.mean(np.abs(np.array(shap_values)), axis=0)

mean_shap = global_temporal_importance(explainer, X_test[:100])
```

**Key Observation**:

-   Almost all SHAP values are positive
    
-   Negative impacts only appear in first timestep

**Key Patterns**:

-   Features show consistent positive values across time
    
-   No strong temporal decay pattern observed

### Why Mostly Positive Impacts?

1.  **Additive Nature of Energy Use**:
    
    -   More features active → Higher energy consumption
        
    -   No competitive "saving" relationships between features
2.  **Data Characteristics**:
    
    -   No strong inverse correlations in training data
        
    -   Minimum energy use = 0 (no negative consumption)

## 5. Conclusion

This analysis reveals an unusually consistent positive-impact pattern, suggesting:

1.  The LSTM learns purely additive relationships
    
2.  Negative impacts are limited to initialization effects
    
3.  Feature selection could safely remove low-impact sensors