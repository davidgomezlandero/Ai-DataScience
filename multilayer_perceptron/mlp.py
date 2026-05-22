import matplotlib.pyplot as pl
import numpy as np
import sys
import pandas as pd
import os
import copy
from functions import softmax, tanh, tanh_derivative, categorical_crossentropy
from tools import open_dataset, networks_conf, standardize, check_balance, calculate_metrics
from models import MLP

def parse():
    options = ["--dataset", "--training", "--predict"]
    if len(sys.argv) < 2 or sys.argv[1] not in options:
        raise Exception("""wrong usage, this program has three options:
    1. \'--dataset\' -> It will take the second argument (.csv) and split in train.csv and test.csv.
    2. \'--training\' -> It will train the model using train.csv.You can run multiple train in one execution with the next structure:
    --training --layer (min 2 hidden layers with at least 2 neurons)num num num... --epochs num(at least 10) --loss (categoricalCrossentropy) --batch_size num(1,455) --learning_rate num(0, 1] optimization(choose between normal(GD) or Adam)).Each time you write from --layer to optimization, you will train a new neural network.
    3. \'--predict\' -> It will make predictions using test.csv by default or the third argument(.csv) and put all them in a results.csv file""")
    
    option = sys.argv[1]
    if option == '--dataset':
        if len(sys.argv) != 3:
            raise Exception("--dataset <dataset.csv>")
       
        if not os.path.exists(sys.argv[2]):
            raise Exception("could not open the .csv file")
    elif option == '--training':
        if not os.path.exists('datasets/data_test.csv'):
            raise Exception("could not open datasets/data_test.csv")
        if not os.path.exists('datasets/data_training.csv'):
            raise Exception("could not open datasets/data_training.csv")
        if len(networks_conf()) == 0:
            raise Exception("wrong parameters input for training")
        
        networks = networks_conf()
        for network in networks:
            if network[0] != '--layer':
                raise Exception(f"wrong starting network parameter: {network[0]}")
            
            i = 1
            while i < len(network) and network[i].isdigit():
                neurons = int(network[i])
                if neurons > 24 or neurons < 2:
                    raise Exception("each layer must have between 2 and 24 neurons")
                i += 1
            num_layers = i - 1
            if num_layers < 2 or num_layers > 10:
                raise Exception("each network must have between 2 and 10 hidden layers")
            if len(network) - i != 10:
                raise Exception("Each network must define exactly all 5 remaining parameters in order (--epochs, --loss, --batch_size, --learning_rate, --optimization)")
            if network[i] != '--epochs':
                raise Exception(f"Expected '--epochs', got '{network[i]}'")
            if not network[i+1].isdigit() or int(network[i+1]) < 10:
                raise Exception("--epochs must be an integer of at least 10")
            i += 2
            if network[i] != '--loss':
                raise Exception(f"Expected '--loss', got '{network[i]}'")
            if network[i+1] not in ['categoricalCrossentropy']:
                raise Exception("--loss must be 'categoricalCrossentropy'")
            i += 2
            if network[i] != '--batch_size':
                 raise Exception(f"Expected '--batch_size', got '{network[i]}'")
            if not network[i+1].isdigit() or int(network[i+1]) < 1 or int(network[i+1]) > 455:
                 raise Exception("--batch_size must be between 1 and 455")
            i += 2
            if network[i] != '--learning_rate':
                raise Exception(f"Expected '--learning_rate', got '{network[i]}'")
            try:
                lr = float(network[i+1])
                if lr <= 0.0 or lr > 1.0:
                     raise Exception("--learning_rate must be between 0.0000001 and 1.0")
            except ValueError:
                 raise Exception("--learning_rate must be a number")
            i += 2
            if network[i] != '--optimization':
                 raise Exception(f"Expected '--optimization', got '{network[i]}'")
            if network[i+1] not in ['GD','Adam']:
                 raise Exception("--optimization must be one of: GD or Adam")
            i += 2
            
    elif option == '--predict':
        predict_file = sys.argv[2] if len(sys.argv) == 3 else 'datasets/data_test.csv'
        if len(sys.argv) > 3:
            raise Exception("the predict option must only have an argument: .csv")
        if not os.path.exists(predict_file):
            raise Exception(f"could not open {predict_file}")
        if not os.path.exists("models.npz"):
            raise Exception("could not open models.npz. You must run --training first.")

def split_dataset():
    raw_data = open_dataset(sys.argv[2])
    # Split 80/20
    raw_data_clean = raw_data.dropna()
    train_data = raw_data_clean.groupby(1, group_keys=False).sample(frac=0.8, random_state=42)
    test_data = raw_data_clean.drop(train_data.index)
    
    train_data = train_data.sample(frac=1, random_state=42)
    test_data = test_data.sample(frac=1, random_state=42)
    
    train_data.to_csv("datasets/data_training.csv", index=False, header=False)
    test_data.to_csv("datasets/data_test.csv", index=False, header=False)

def training():
    # 1. Load proper Train and Test sets instead of internally splitting train.csv
    train_data = open_dataset('datasets/data_training.csv')
    test_data = open_dataset('datasets/data_test.csv')
    if not os.path.exists('curves'):
        os.makedirs('curves')
    if not os.path.exists('history_models'):
        os.makedirs('history_models')
    
    
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Process Training Data
    X_train = train_data.iloc[:, 2:].values.astype(np.float64)
    Y_train_dummies = pd.get_dummies(train_data.iloc[:, 1])
    class_labels = Y_train_dummies.columns.values.astype(str)
    Y_train = Y_train_dummies.values.astype(np.float64)
    
    # Process Validation Data (test.csv)
    X_val = test_data.iloc[:, 2:].values.astype(np.float64)
    # Reindex ensures test.csv gets the exact same M/B columns even if one is missing in a small dataset
    Y_val_dummies = pd.get_dummies(test_data.iloc[:, 1]).reindex(columns=Y_train_dummies.columns, fill_value=0)
    Y_val = Y_val_dummies.values.astype(np.float64)
    
    # Normalization
    X_train, feature_means, feature_stds = standardize(X_train)
    X_val, _, _ = standardize(X_val, mean=feature_means, std=feature_stds)
    
    input_size = X_train.shape[1]
    output_size = Y_train.shape[1]
    networks = networks_conf()
    models_to_save = {'classes': class_labels}
    
    for net_idx, network in enumerate(networks):
        i = 1
        hidden_neurons = []
        while network[i].isdigit():
            hidden_neurons.append(int(network[i]))
            i += 1
            
        epochs, batch_size = int(network[i+1]), int(network[i+5])
        base_lr = float(network[i+7])
        opt_choice = network[i+9]
        layer_sizes = [input_size] + hidden_neurons + [output_size]
        
        actual_batch_size = batch_size if batch_size <= len(X_train) else len(X_train)
        
        lr_variations = [base_lr, base_lr / 10.0, base_lr * 10.0]
        
        # We will create 2 side-by-side graphs: One for Loss, One for Accuracy
        fig, (ax1, ax2) = pl.subplots(1, 2, figsize=(15, 6))
        
        best_overall_model = None
        best_overall_val_loss = float('inf')
        
        print(f"=========================================")
        print(f"Testing Config {net_idx} | Architecture: {layer_sizes} | Opt: {opt_choice}")
        print(f"Training Data: {len(X_train)} rows | Validation Data (Test.csv): {len(X_val)} rows")
        
        for v_idx, current_lr in enumerate(lr_variations):
            print(f"-- Variant {v_idx+1}: Learning Rate = {current_lr:.5f} --")
            model = MLP(layer_sizes)
            history = []
            
            best_val_loss = float('inf')
            patience_counter = 0
            PATIENCE_LIMIT = 15
            
            for epoch in range(epochs):
                # Mini-Batch Iteration
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                
                for start_idx in range(0, X_train.shape[0], actual_batch_size):
                    end_idx = min(start_idx + actual_batch_size, X_train.shape[0])
                    batch_idx = indices[start_idx:end_idx]
                    X_batch, Y_batch = X_train[batch_idx], Y_train[batch_idx]
                    
                    y_pred_batch = model.forward(X_batch)
                    model.backward(Y_batch, y_pred_batch, current_lr, opt_choice)
                
                # Full evaluation at end of epoch on Train and Val datasets
                y_train_pred = model.forward(X_train)
                train_loss = categorical_crossentropy(Y_train, y_train_pred)
                train_acc, _, _, _ = calculate_metrics(Y_train, y_train_pred)
                
                y_val_pred = model.forward(X_val)
                val_loss = categorical_crossentropy(Y_val, y_val_pred)
                val_acc, val_prec, val_rec, val_f1 = calculate_metrics(Y_val, y_val_pred)
                
                history.append({
                    'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': val_loss, 'val_acc': val_acc, 
                    'val_prec': val_prec, 'val_rec': val_rec, 'val_f1': val_f1
                })
                
                # Early Stopping Logic (Monitoring Validation Loss on test.csv)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if val_loss < best_overall_val_loss:
                        best_overall_val_loss = val_loss
                        best_overall_model = copy.deepcopy(model)
                else:
                    patience_counter += 1
                    
                print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
                    
                if patience_counter >= PATIENCE_LIMIT:
                    print(f"Early stopping triggered at Epoch {epoch}!")
                    break

            # Save Metrics CSV history
            history_df = pd.DataFrame(history)
            history_df.to_csv(f'history_models/history_model_{net_idx}_lr_{current_lr:.5f}.csv', index=False)
            
            # Plot Loss Curves
            ax1.plot(history_df['epoch'], history_df['train_loss'], label=f'Train LR {current_lr:.5f}')
            ax1.plot(history_df['epoch'], history_df['val_loss'], '--', label=f'Val LR {current_lr:.5f}')
            
            # Plot Accuracy Curves
            ax2.plot(history_df['epoch'], history_df['train_acc'], label=f'Train LR {current_lr:.5f}')
            ax2.plot(history_df['epoch'], history_df['val_acc'], '--', label=f'Val LR {current_lr:.5f}')
        
        # Graph styling
        ax1.set_title(f'Loss Curves (Model {net_idx})')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title(f'Accuracy Curves (Model {net_idx})')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        arch_str = "-".join(map(str, layer_sizes))
        plot_filename = f"curves/curves_arch-{arch_str}_opt-{opt_choice}_epochs-{epochs}_bs-{actual_batch_size}_lr-{base_lr}.png"
        
        pl.tight_layout()
        pl.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}\n")
        pl.close()
        
        # Extract Best Global Configuration out of the variations
        models_to_save[f'model_{net_idx}_num_layers'] = len(best_overall_model.layers)
        for l_idx, layer in enumerate(best_overall_model.layers):
            models_to_save[f'model_{net_idx}_layer_{l_idx}_W'] = layer.W
            models_to_save[f'model_{net_idx}_layer_{l_idx}_b'] = layer.b
            
    models_to_save['classes'] = class_labels
    models_to_save['feature_means'] = feature_means
    models_to_save['feature_stds'] = feature_stds 

    np.savez('models.npz', **models_to_save)
    print(f"Training completed! All best configurations saved to models.npz")

def predict():
    predict_file = sys.argv[2] if len(sys.argv) == 3 else 'datasets/data_test.csv'
    print(f"Making predictions on {predict_file}...")
    test_data = open_dataset(predict_file)
    X_test = test_data.iloc[:, 2:].values.astype(np.float64)
    Y_test = pd.get_dummies(test_data.iloc[:, 1])
    
    saved_params = np.load('models.npz')
    class_labels = saved_params['classes']
    
    feature_means = saved_params['feature_means']
    feature_stds = saved_params['feature_stds']
    X_test, _, _ = standardize(X_test, mean=feature_means, std=feature_stds)
    
    model_indices = set([int(k.split('_')[1]) for k in saved_params.files if k.startswith('model_')])
    
    results_df = pd.DataFrame()
    for net_idx in sorted(model_indices):
        num_layers = saved_params[f'model_{net_idx}_num_layers']
        A = X_test
        for l_idx in range(num_layers):
            W = saved_params[f'model_{net_idx}_layer_{l_idx}_W']
            b = saved_params[f'model_{net_idx}_layer_{l_idx}_b']
            Z = np.dot(A, W) + b
            A = softmax(Z) if l_idx == num_layers - 1 else tanh(Z)
            
        predictions = class_labels[np.argmax(A, axis=1)]
        results_df[f'Model_{net_idx}_Predictions'] = predictions
        loss = categorical_crossentropy(Y_test, A)
        print(f"Evaluation Loss for Model ({net_idx}): {loss:.4f}")
    
    results_df.to_csv('results.csv', index=False)
    print("Predictions saved to results.csv!")

if __name__ == '__main__':
    try:
        parse()
        option = sys.argv[1]
        
        if option == '--dataset':
            split_dataset()
            print("Dataset successfully split into train.csv and test.csv")
            check_balance(open_dataset('test.csv'))
            check_balance(open_dataset('train.csv'))
            
        elif option == '--training':
            training()
            
        elif option == '--predict':
            predict()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)