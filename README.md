# Création venv
python -m venv .venv

# Activation venv
source .venv/bin/activate

# Installation dépendances
pip install -r requirements.txt

# Run
python main.py
mlflow ui
