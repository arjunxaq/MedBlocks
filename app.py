from flask import Flask, jsonify, render_template
import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
import requests
from web3 import Web3
import os
from dotenv import load_dotenv

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')

# -------------------------
# CONFIG
# -------------------------
load_dotenv()  # loads .env file

contract_json_str = os.environ.get("CONTRACT_JSON")
contract = json.loads(contract_json_str)
INFURA_URL = os.environ.get("INFURA_KEY")
with open(contract) as f:
    contract_info = json.load(f)

w3 = Web3(Web3.HTTPProvider(INFURA_URL))
contract = w3.eth.contract(address=contract_info["address"], abi=contract_info["abi"])

# -------------------------
# Helper Functions
# -------------------------
def download_from_ipfs(cid, filename):
    url = f"https://gateway.pinata.cloud/ipfs/{cid}"
    r = requests.get(url)
    if r.status_code == 200:
        with open(filename, "wb") as f:
            f.write(r.content)
    else:
        raise Exception(f"Failed to fetch {cid}")

def fetch_ipfs_hashes():
    hospitals = contract.functions.getAllHospitals().call()
    models, stats = [], []

    for h in hospitals:
        latest = contract.functions.getLatestModelData(h).call()
        model_cid, stats_cid = latest[0], latest[1]

        model_file = f"{h}_model.pth"
        stats_file = f"{h}_stats.json"
        download_from_ipfs(model_cid, model_file)
        download_from_ipfs(stats_cid, stats_file)

        models.append(model_file)
        stats.append(stats_file)

    return models, stats

# -------------------------
# Optimized GAN Generator
# -------------------------
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # assume data scaled to [-1,1]
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/wallet')
def wallet():
    return render_template('wallet.html')

@app.route('/query')
def query():
    return render_template('query.html')

@app.route('/generate', methods=['GET'])
def generate_data():
    try:
        org_models, org_stats = fetch_ipfs_hashes()

        # Load column stats from first hospital (assume same schema)
        with open(org_stats[0], "r") as f:
            col_stats = json.load(f)
        cols = list(col_stats.keys())
        data_dim = len(cols)
        latent_dim = 32  # increased for better GAN capacity

        # Load each generator from IPFS
        generators = []
        for model_file in org_models:
            g = Generator(latent_dim, data_dim)
            g.load_state_dict(torch.load(model_file, map_location='cpu'))
            g.eval()
            generators.append(g)

        # Aggregate weights to create master generator
        master_generator = Generator(latent_dim, data_dim)
        state_dict = master_generator.state_dict()
        for key in state_dict.keys():
            weights = [g.state_dict()[key] for g in generators]
            state_dict[key] = sum(weights) / len(weights)
        master_generator.load_state_dict(state_dict)
        master_generator.eval()

        # Generate synthetic data
        num_samples = 100  # increased for better stats
        z = torch.randn(num_samples, latent_dim)
        synthetic_data = master_generator(z).detach().numpy()

        # Rescale synthetic data to original column ranges
        synthetic_rescaled = []
        for i, col in enumerate(cols):
            col_min = col_stats[col]["min"]
            col_max = col_stats[col]["max"]
            vals = 0.5 * (synthetic_data[:, i] + 1) * (col_max - col_min) + col_min
            synthetic_rescaled.append(vals)
        synthetic_rescaled = np.array(synthetic_rescaled).T

        # Save CSV
        df_synth = pd.DataFrame(synthetic_rescaled, columns=cols)
        csv_path = "synthetic_data.csv"
        df_synth.to_csv(csv_path, index=False)

        return jsonify(df_synth.to_dict(orient="records"))

    except Exception as e:
        import traceback
        print("🔥 ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/get_synthetic_data', methods=['GET'])
def get_synthetic_data():
    try:
        csv_path = "synthetic_data.csv"
        if not os.path.exists(csv_path):
            return jsonify({"error": "File not ready"}), 404
        df = pd.read_csv(csv_path)
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_model_history', methods=['GET'])
def get_model_history():
    try:
        hospitals = contract.functions.getAllHospitals().call()
        history = []

        for h in hospitals:
            models = contract.functions.getModelData(h).call()
            model_versions = []
            for idx, model in enumerate(models):
                model_versions.append({
                    "version": idx + 1,
                    "modelHash": model[0],
                    "statsHash": model[1]
                })
            history.append({
                "hospital": h,
                "models": model_versions
            })

        return jsonify(history)
    except Exception as e:
        import traceback
        print("🔥 ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
