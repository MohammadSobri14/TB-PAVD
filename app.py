from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open('Notebook/apriori_model.pkl', 'rb') as f:
    apriori_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request")
    try:
        data = request.json
        transactions = data.get('transactions')
        print("Transactions received:", transactions)
        if not transactions:
            return jsonify({"error": "No transactions provided"}), 400
        results = apriori_model.run(transactions)
        print("Results:", results)
        return jsonify(results)
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
