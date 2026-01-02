from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd



application = Flask(__name__)
app = application

# Load the trained model, scaler, and cluster mapping
model = joblib.load('./model/iris_model.pkl')
scaler = joblib.load('./model/scaler.pkl')
cluster_map = joblib.load('./model/cluster_map.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Capture the form data
            data = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]

            # 2. Create a DataFrame with the EXACT names used during training
            # Note: If your scaler used different names, use those instead
            feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            features_df = pd.DataFrame([data], columns=feature_names)

            # 3. Scale using the DataFrame (This stops the warning)
            scaled_features = scaler.transform(features_df)

            # 4. Predict
            cluster_label = model.predict(scaled_features)[0]
            
            # 5. Map to identity (Ensure keys in cluster_map are integers)
            flower_identity = cluster_map.get(int(cluster_label), "Unknown")

            return render_template('index.html', prediction=flower_identity)
            
        except Exception as e:
            return render_template('index.html', prediction=f"Error: {e}")
    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)