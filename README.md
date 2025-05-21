# Création venv
python -m venv .venv

# Activatio venv
source .venv/bin/activate

# Installation dépendances
pip install -r requirements.txt

# Run
python main.py
mlflow ui
