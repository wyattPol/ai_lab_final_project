import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import random

class WandBConfig:
    def __init__(self):
        self.api_key = "a0fc75f04fa27bc24039cf264e6500367853626f"
        self.project_name = "ai_project"
        
    def setup(self):
        os.environ["WANDB_API_KEY"] = self.api_key
        wandb.init(project=self.project_name)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BasicNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BasicNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.conv_size = int(np.sqrt(input_size)) 
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)
        )
        
        self.fc_size = 32 * input_size
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        x = x.view(-1, 1, self.input_size)
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_size)
        x = self.fc_layers(x)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(1, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1, 1)
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class MLProject:
    def __init__(self, model_type='basic', config=None):
        self.config = config or {
            'n_samples': 1000,
            'n_features': 20,
            'n_classes': 3,
            'n_informative': 4,
            'hidden_size': 64,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'train_split': 0.7,    
            'val_split': 0.15,    
            'test_split': 0.15,    
            'model_type': model_type
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None
        self.model_type = model_type
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def generate_data(self):
        X, y = make_classification(
            n_samples=self.config['n_samples'],
            n_features=self.config['n_features'],
            n_classes=self.config['n_classes'],
            n_informative=self.config['n_informative'],
            n_redundant=int(self.config['n_features'] * 0.1),
            n_clusters_per_class=2,
            n_repeated=0,
            class_sep=2.0,  
            flip_y=0.01,    
            random_state=42
        )
        
        # visualize generated data using first two informative features
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='deep')
        plt.title('Generated Classification Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.savefig('data_visualization.png')
        wandb.log({"data_visualization": wandb.Image('data_visualization.png')})
        plt.close()
        
        return X, y
    
    def prepare_data(self, X, y):
        """split the dataset into train, validation, and test sets"""

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config['test_split'],
            stratify=y,
            random_state=42
        )
        
        val_size = self.config['val_split'] / (1 - self.config['test_split'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp,
            random_state=42
        )
        # scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        train_dataset = CustomDataset(X_train, y_train)
        val_dataset = CustomDataset(X_val, y_val)
        test_dataset = CustomDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'])
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        return train_loader, val_loader, test_loader
    
    def build_model(self):
        model_classes = {
            'basic': BasicNN,
            'cnn': CNN,
            'lstm': LSTM
        }
        
        model_class = model_classes.get(self.model_type)
        if model_class is None:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = model_class(
            input_size=self.config['n_features'],
            hidden_size=self.config['hidden_size'],
            num_classes=self.config['n_classes']
        ).to(self.device)
        
        return self.model
    
    def train_model(self, train_loader, val_loader):
        """Train the model with validation"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        best_val_accuracy = 0
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # train metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * train_correct / train_total
            
            # validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # vvalidation metrics
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100. * val_correct / val_total
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            print(f'Epoch: {epoch}')
            print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
    
    def evaluate_model(self, test_loader):
        """evaluate the model on test data"""
        self.model.eval()
        correct = 0
        total = 0
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        if len(predictions) > 0:
            self.plot_confusion_matrix(true_labels, predictions)
        
        return 100. * correct / total
    
    def plot_confusion_matrix(self, true_labels, predictions):
        """Plot and log confusion matrix"""
        cm = np.zeros((self.config['n_classes'], self.config['n_classes']))
        for t, p in zip(true_labels, predictions):
            cm[t][p] += 1
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})
        plt.close()
    
    
    def predict(self, input_data):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)
            
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
            
        scaled_data = self.scaler.transform(input_data)
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(scaled_data).to(self.device)
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
        return predicted.cpu().numpy(), probabilities.cpu().numpy()

    def get_user_input_and_predict(self):

        choice = input("Would you like to enter features manually or generate 20 random numbers? (Enter 'manual' or 'auto'): ").strip().lower()
        
        if choice == 'manual':
            print("Please enter the features separated by spaces:")
            user_input = input()  
            input_data = np.array([float(x) for x in user_input.split()])  

            if input_data.shape[0] != self.config['n_features']:
                raise ValueError(f"Expected {self.config['n_features']} features, but received {input_data.shape[0]}.")

        elif choice == 'auto':
            # generate 20 random numbers between 1 and 100
            input_data = np.array([random.randint(1, 100) for _ in range(20)])
            print(f"Generated 20 random features: {input_data}")

        else:
            print("Invalid choice. Please enter 'manual' or 'auto'.")
            return

        scaled_input = self.scaler.transform(input_data.reshape(1, -1))

        predicted_class, probabilities = self.predict(scaled_input)

        print(f"Predicted class: {predicted_class}")
        print(f"Class probabilities: {probabilities}")

        return predicted_class, probabilities

def plot_model_comparison(results_dict):
    plt.figure(figsize=(15, 10))
    
    # plot losses
    plt.subplot(2, 1, 1)
    for model_type, history in results_dict.items():
        epochs = range(1, len(history['train_losses']) + 1)
        plt.plot(epochs, history['train_losses'], '-', label=f'{model_type} Train')
        plt.plot(epochs, history['val_losses'], '--', label=f'{model_type} Val')
    
    plt.title('Training and Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # plot accuracies
    plt.subplot(2, 1, 2)
    for model_type, history in results_dict.items():
        epochs = range(1, len(history['train_accuracies']) + 1)
        plt.plot(epochs, history['train_accuracies'], '-', label=f'{model_type} Train')
        plt.plot(epochs, history['val_accuracies'], '--', label=f'{model_type} Val')
    
    plt.title('Training and Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    wandb.log({"model_comparison": wandb.Image('model_comparison.png')})
    plt.close()

def main():
    wandb_config = WandBConfig()
    wandb_config.setup()
    model_types = ['basic', 'cnn', 'lstm']
    config = {
        'n_samples': 2000,
        'n_features': 20,
        'n_classes': 3,
        'n_informative': 4,
        'hidden_size': 64,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'train_split': 0.7,
        'val_split': 0.2,
        'test_split': 0.1
    }

    all_results = {}
    
    # train 
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model:")
        project = MLProject(model_type=model_type, config=config)
    
        X, y = project.generate_data()
        train_loader, val_loader, test_loader = project.prepare_data(X, y)
        model = project.build_model()
        project.train_model(train_loader, val_loader)
        
        # evaluate
        test_accuracy = project.evaluate_model(test_loader)
        print(f"{model_type.upper()} Test Accuracy: {test_accuracy:.2f}%")
    
        all_results[model_type] = {
            'train_losses': project.train_losses,
            'val_losses': project.val_losses,
            'train_accuracies': project.train_accuracies,
            'val_accuracies': project.val_accuracies
        }

    plot_model_comparison(all_results)
    print("\nFinal Model Comparisons:")
    print("-" * 50)
    for model_type, results in all_results.items():
        final_train_loss = results['train_losses'][-1]
        final_val_loss = results['val_losses'][-1]
        final_train_acc = results['train_accuracies'][-1]
        final_val_acc = results['val_accuracies'][-1]
        
        print(f"\n{model_type.upper()} Model:")
        print(f"Final Training Loss: {final_train_loss:.4f}")
        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    
    predicted_class, probabilities = project.get_user_input_and_predict()
    print(f"User Input Prediction: Class {predicted_class}, Probabilities {probabilities}")
    wandb.finish()

if __name__ == "__main__":
    main()
