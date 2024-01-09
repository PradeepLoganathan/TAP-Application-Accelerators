from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model (make sure the path to 'finalized_model.sav' is correct)
with open(PICKLE_FILE, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the input feature from the request
        data = request.json
        input_feature = [[data['feature']]]

        # Make a prediction
        prediction = model.predict(input_feature)

        # Send back the prediction
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
