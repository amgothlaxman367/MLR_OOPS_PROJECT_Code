🏠 Multiple Linear Regression using OOP in Python
📌 Project Overview

This project demonstrates the implementation of Multiple Linear Regression (MLR) using Object-Oriented Programming (OOP) in Python.
The model is trained to predict house prices based on various features like bedrooms, bathrooms, square footage, and more.

🎯 Objectives
Implement MLR using OOP concepts
Perform data preprocessing (encoding categorical data)
Train and evaluate the model
Predict new/custom input data
Save and load trained model using pickle
🛠️ Technologies Used
Python 🐍
Pandas
NumPy
Scikit-learn
Pickle
📂 Project Structure
├── data (1).csv        # Dataset
├── model.py            # Main Python script (MLR class)
├── MODEL.pkl           # Saved model file
├── README.md           # Project documentation
⚙️ Features

✔️ OOP-based design
✔️ Data preprocessing (city & country encoding)
✔️ Train-test split
✔️ Model training using Linear Regression
✔️ Evaluation using R² Score and RMSE
✔️ Custom prediction support
✔️ Model saving & loading

🧠 Model Workflow
Load dataset
Encode categorical features
Split data into training and testing sets
Train Linear Regression model
Evaluate performance
Predict new data
Save model using pickle
📊 Evaluation Metrics
R² Score (Accuracy)
Root Mean Squared Error (RMSE)
▶️ How to Run
Clone the repository
git clone https://github.com/your-username/your-repo-name.git
Navigate to project folder
cd your-repo-name
Install dependencies
pip install pandas numpy scikit-learn
Run the script
python model.py
🧪 Sample Prediction

The model predicts house price based on inputs like:

Bedrooms
Bathrooms
Square footage
Floors
Year built
City & Country
💾 Model Saving

The trained model is saved as:

MODEL.pkl

It can be reused without retraining.

🚀 Future Improvements
Add proper encoding (One-Hot Encoding)
Use advanced models (Random Forest, XGBoost)
Add visualization
Deploy as web app
👨‍💻 Author

Amgoth Laxman
Data Science & Machine Learning Enthusiast

📌 Note

This project is created for learning and educational purposes.
