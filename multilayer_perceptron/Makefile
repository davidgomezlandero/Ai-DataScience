VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

all: $(VENV)/bin/activate
	@echo "--------------------------------------------------------"
	@echo " Makefile Usage Instructions:                           "
	@echo "--------------------------------------------------------"
	@echo " make clean     : Deletes datasets, curves/, and history_models/."
	@echo " make fclean    : Runs 'clean' and deletes models.npz & results.csv."
	@echo " make training1 : Runs Training Example A from the README."
	@echo " make training2 : Runs Training Example B from the README."
	@echo "--------------------------------------------------------"
	@echo " ⚠️  Virtual Environment Activation:                      "
	@echo " 'make' cannot change your current terminal's environment."
	@echo " To activate the venv manually, run this command:       "
	@echo "                                                        "
	@echo " source $(VENV)/bin/activate"
	@echo "--------------------------------------------------------"
	@echo " 🚀 Project Options (Manual CLI Usage):                 "
	@echo "--------------------------------------------------------"
	@echo " 1. Splitting the Dataset:"
	@echo "    python mlp.py --dataset data.csv"
	@echo ""
	@echo " 2. Training the Network (Example):"
	@echo "    python mlp.py --training --layer 24 24 --epochs 100 \\"
	@echo "           --loss categoricalCrossentropy --batch_size 32 \\"
	@echo "           --learning_rate 0.01 --optimization Adam"
	@echo ""
	@echo " 3. Making Predictions:"
	@echo "    python mlp.py --predict"
	@echo "    python mlp.py --predict test.csv"
	@echo "--------------------------------------------------------"

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

training1: $(VENV)/bin/activate
	$(PYTHON) mlp.py --training --layer 24 24 --epochs 100 --loss categoricalCrossentropy --batch_size 32 --learning_rate 0.01 --optimization Adam

training2: $(VENV)/bin/activate
	$(PYTHON) mlp.py --training \
    --layer 5 5 --epochs 150 --loss categoricalCrossentropy --batch_size 16 --learning_rate 0.005 --optimization GD \
    --layer 16 16 --epochs 200 --loss categoricalCrossentropy --batch_size 32 --learning_rate 0.01 --optimization Adam

clean:
	@echo "Cleaning up generated assets..."
	@rm -f datasets/data_training.csv datasets/data_test.csv
	@rm -f train.csv test.csv
	@rm -rf curves/
	@rm -rf history_models/
	@rm -rf __pycache__

fclean: clean
	@echo "Performing deep clean (deleting models and results)..."
	@rm -f models.npz
	@rm -f results.csv

re: fclean all

.PHONY: all clean fclean training1 training2