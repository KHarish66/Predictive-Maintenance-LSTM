# Predictive-Maintenance-LSTM

Predictive Maintenance for Industrial Machinery using Deep Learning
A deep learning solution to forecast equipment failure and optimize maintenance schedules.
Live Application Demo: https://predictive-maintenance-lstm-abre9dtxwjbrvjf87fjhyk.streamlit.app/ 🚀

Overview
This project demonstrates an end-to-end data science workflow for predictive maintenance. It utilizes a Long Short-Term Memory (LSTM) deep learning model to predict the Remaining Useful Life (RUL) of complex machinery based on time-series sensor data. The final, high-performance model is deployed as an interactive web application using Streamlit, allowing for real-time health assessment.

While the model was trained on a benchmark dataset for aircraft turbofan engines, the methodology is directly applicable to any industrial equipment that generates sensor data, such as the machinery used in modern tyre manufacturing.

Business Problem : 
In capital-intensive industries like manufacturing and aviation, unexpected equipment failure is a critical issue. It leads to:

Unplanned Downtime: Halting production lines and causing significant financial losses.

High Repair Costs: Emergency repairs are far more expensive than scheduled maintenance.

Safety Risks: A failing machine can pose a danger to personnel.

This project solves this by creating a proactive system that forecasts failures before they happen, enabling a data-driven, cost-effective maintenance strategy.

Technical Workflow
The project followed a structured data science methodology:

Data Preprocessing & EDA: Loaded and cleaned the NASA C-MAPSS time-series dataset. Visualized sensor trends to confirm patterns of degradation over time.

Feature Engineering: Calculated the Remaining Useful Life (RUL) for each engine in the training set. This crucial step created the ground-truth target variable for the regression model.

Feature Scaling: Applied Min-Max normalization to all sensor and setting features, scaling them to a range of [0, 1] for optimal model performance.

Data Sequencing: Transformed the time-series data into overlapping sequences (or "sliding windows") of 30 cycles. This provided the necessary historical context for the LSTM model to learn from trends.

Model Development: Built, trained, and optimized an LSTM neural network using TensorFlow and Keras. The model was trained to predict the RUL based on the input sequences.

Evaluation: The final model was rigorously evaluated on an unseen test set to measure its real-world predictive accuracy.

Final Results
The final model demonstrated strong performance, proving its ability to accurately forecast the remaining life of the machinery.

Metric

Score

Description

RMSE (Root Mean Squared Error)

17.85 cycles

The model's predictions are, on average, off by only ~19 operational cycles.

R-squared (R²) Score

0.82

The model successfully explains 82% of the variability in the engine's RUL.

How to Use the Live Application
Navigate to the live application URL.

Use the sidebar to input the 24 required operational settings and sensor values for a piece of equipment.

Click the "Predict" button.

The application will display the predicted Remaining Useful Life in cycles and a corresponding health status (Good, Warning, or Critical).

Technologies Used
Core Programming: Python

Data Manipulation & Analysis: Pandas, NumPy

Machine Learning & Preprocessing: Scikit-learn

Deep Learning: TensorFlow,
