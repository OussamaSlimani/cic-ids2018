from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
loaded_cnn_model = load_model('cnn_model.h5')
print("Model loaded successfully.")

# Define class mapping
class_mapping = {
    0: "Benign",
    1: "Bot",
    2: "Brute Force -Web",
    3: "Brute Force -XSS",
    4: "DDOS attack-HOIC",
    5: "DDOS attack-LOIC-UDP",
    6: "DDoS attacks-LOIC-HTTP",
    7: "DoS attacks-GoldenEye",
    8: "DoS attacks-Hulk",
    9: "DoS attacks-SlowHTTPTest",
    10: "DoS attacks-Slowloris",
    11: "FTP-BruteForce",
    12: "Infilteration",
    13: "SQL Injection",
    14: "SSH-Bruteforce"
}

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Parse the JSON input data
    input_data = request.get_json()

    try:
        # Convert JSON to DataFrame
        input_df = pd.DataFrame(input_data)

        # Reshape the input for the CNN model
        input_cnn = input_df.values.reshape(1, input_df.shape[1], 1)  # 1 sample, N features, 1 channel

        # Make the prediction
        prediction = loaded_cnn_model.predict(input_cnn)
        predicted_class_id = int(np.argmax(prediction, axis=1)[0])  # Convert to Python int for JSON serialization
        predicted_class = class_mapping.get(predicted_class_id, "Unknown")

        # Return the prediction as JSON
        return jsonify({
            'class_id': predicted_class_id,
            'predicted_class': predicted_class
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
