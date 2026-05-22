import matplotlib.pyplot as pl
import numpy as np
import sys
import pandas as pd
from os import path
import copy

def open_dataset(dataset):
    data = pd.read_csv(dataset, header=None)
    return data

def networks_conf():
    args = sys.argv[2:]
    networks = []
    current_network = []
    
    for arg in args:
        if arg == '--layer' and current_network:
            networks.append(current_network)
            current_network = [arg]
        else:
            current_network.append(arg)
    if current_network:
        networks.append(current_network)
    
    return networks    

def standardize(X, mean=None, std=None):
    """
    Applies Z-score normalization. 
    If mean and std are not provided, it calculates them from the dataset X.
    """
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Prevent division by zero for columns that have zero variance
        std = np.where(std == 0, 1e-15, std)
        
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

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
       
        if not path.exists(sys.argv[2]):
            raise Exception("could not open the .csv file")
    elif option == '--training':
        if not path.exists('test.csv'):
            raise Exception("could not open test.csv")
        if not path.exists('train.csv'):
            raise Exception("could not open train.csv")
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
        predict_file = sys.argv[2] if len(sys.argv) == 3 else 'test.csv'
        if len(sys.argv) > 3:
            raise Exception("the predict option must only have an argument: .csv")
        if not path.exists(predict_file):
            raise Exception(f"could not open {predict_file}")
        if not path.exists("models.npz"):
            raise Exception("could not open models.npz. You must run --training first.")

def split_dataset():
    raw_data = open_dataset(sys.argv[2])
    # Split 80/20
    raw_data_clean = raw_data.dropna()
    train_data = raw_data_clean.groupby(1, group_keys=False).sample(frac=0.8, random_state=42)
    test_data = raw_data_clean.drop(train_data.index)
    
    train_data = train_data.sample(frac=1, random_state=42)
    test_data = test_data.sample(frac=1, random_state=42)
    
    train_data.to_csv("train.csv", index=False, header=False)
    test_data.to_csv("test.csv", index=False, header=False)

def check_balance(dataset):
    class_counts = dataset[1].value_counts()
    print("Class distribution:")
    print(class_counts)
    print("\nPercentages:")
    print(class_counts / len(dataset) * 100)
    print()

# --- Math & Metrics ---
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(a):
    return 1.0 - np.power(a, 2)

def categorical_crossentropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def calculate_metrics(y_true, y_pred):
    y_t = np.argmax(y_true, axis=1)
    y_p = np.argmax(y_pred, axis=1)
    
    classes = np.unique(np.concatenate((y_t, y_p)))
    precisions = []
    recalls = []
    f1s = []
    
    for c in classes:
        tp = np.sum((y_p == c) & (y_t == c))
        fp = np.sum((y_p == c) & (y_t != c))
        fn = np.sum((y_p != c) & (y_t == c))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    acc = np.mean(y_t == y_p)
    return acc, np.mean(precisions), np.mean(recalls), np.mean(f1s)

# --- Network Architecture ---
class Layer:
    def __init__(self, nin, nout, is_output=False):
        self.W = np.random.randn(nin, nout) * np.sqrt(2. / nin)
        self.b = np.zeros((1, nout))
        self.is_output = is_output
        self.Inputs, self.Z, self.A = None, None, None
        
        # Adam Parameters
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, inputs):
        self.Inputs = inputs
        self.Z = np.dot(inputs, self.W) + self.b
        self.A = softmax(self.Z) if self.is_output else tanh(self.Z)
        return self.A

class MLP:
    def __init__(self, layer_sizes):
        self.layers = []
        self.t = 0 # Global step for Adam
        for i in range(len(layer_sizes) - 1):
            is_out = (i == len(layer_sizes) - 2)
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], is_output=is_out))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y_true, y_pred, learning_rate, opt):
        m = y_true.shape[0]
        dZ = y_pred - y_true
        self.t += 1
        
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            dW = np.dot(layer.Inputs.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            if i > 0:
                prev_layer = self.layers[i - 1]
                dA = np.dot(dZ, layer.W.T)
                dZ = dA * tanh_derivative(prev_layer.A)
            
            if opt == 'Adam':
                layer.mW = beta1 * layer.mW + (1 - beta1) * dW
                layer.vW = beta2 * layer.vW + (1 - beta2) * (dW ** 2)
                mWh = layer.mW / (1 - beta1 ** self.t)
                vWh = layer.vW / (1 - beta2 ** self.t)
                layer.W -= learning_rate * mWh / (np.sqrt(vWh) + epsilon)
                
                layer.mb = beta1 * layer.mb + (1 - beta1) * db
                layer.vb = beta2 * layer.vb + (1 - beta2) * (db ** 2)
                mbh = layer.mb / (1 - beta1 ** self.t)
                vbh = layer.vb / (1 - beta2 ** self.t)
                layer.b -= learning_rate * mbh / (np.sqrt(vbh) + epsilon)
            else: # Fallback to GD
                layer.W -= learning_rate * dW
                layer.b -= learning_rate * db

def training():
    # 1. Load proper Train and Test sets instead of internally splitting train.csv
    train_data = open_dataset('train.csv')
    test_data = open_dataset('test.csv')
    
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
            history_df.to_csv(f'history_model_{net_idx}_lr_{current_lr:.5f}.csv', index=False)
            
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
        plot_filename = f"curves_arch-{arch_str}_opt-{opt_choice}_epochs-{epochs}_bs-{actual_batch_size}_lr-{base_lr}.png"
        
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
    predict_file = sys.argv[2] if len(sys.argv) == 3 else 'test.csv'
    print(f"Making predictions on {predict_file}...")
    test_data = open_dataset(predict_file)
    X_test = test_data.iloc[:, 2:].values.astype(np.float64)
    
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