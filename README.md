# Comparison of Classification Models for Feature-Based Prediction

This project aims to demonstrate the performance of different machine learning models for classification tasks, using user-provided features or randomly generated features as input. The project evaluates and compares the performance of three different models: a basic classification model, a convolutional neural network (CNN), and a long short-term memory network (LSTM). The goal is to predict a class label based on the given features and display the results with appropriate accuracy metrics and plots.

## Project Structure

### Contents:
- `main.py`: Main script for model training, evaluation, and inference.
- `README.md`: This file, which provides an overview of the project.
- 
### Basic Training parameter Configs:
-  `Epochs`: 100
-  `Learning Rate`: 0.001
-  `Training set size`: 1400 sample data

## Methods

### 1. **Basic Model**
The basic model serves as a baseline for performance comparison. It is a simple fully connected network designed to process the input features and predict the class. This model is lightweight and fast for evaluation.

### 2. **Convolutional Neural Network (CNN)**
The CNN is designed to process feature sequences or structured data by using convolutions. It applies filters to detect patterns, improving its ability to handle spatial or sequential data. CNNs are well known for their powerful feature extraction capabilities, which make them effective for image-like input data.

### 3. **Long Short-Term Memory Network (LSTM)**
The LSTM model is a type of recurrent neural network (RNN) that can handle sequences of data, making it ideal for tasks involving temporal dependencies. Despite being traditionally used in time-series problems, it can also be used here to handle sequential data from features.

## Model Structures

### Basic Model
- **Layers**: Input layer → Fully connected hidden layers → Output layer
- **Activation Function**: ReLU for hidden layers, Softmax for output
- **Loss Function**: Cross-entropy loss
- **Optimizer**: Adam

### CNN Model
- **Layers**: Convolutional layers → Max pooling layers → Fully connected layers → Output layer
- **Activation Function**: ReLU for convolutional and fully connected layers, Softmax for output
- **Loss Function**: Cross-entropy loss
- **Optimizer**: Adam

### LSTM Model
- **Layers**: LSTM layers → Fully connected layers → Output layer
- **Activation Function**: ReLU for hidden layers, Softmax for output
- **Loss Function**: Cross-entropy loss
- **Optimizer**: Adam

## Results

### Test Accuracy:
- **BASIC Model**: 95.50%
- **CNN Model**: 94.50%
- **LSTM Model**: 89.00%

### Final Model Comparisons:

#### BASIC Model:
- Final Training Loss: 0.1194
- Final Validation Loss: 0.1215
- Final Training Accuracy: 96.21%
- Final Validation Accuracy: 96.50%

#### CNN Model:
- Final Training Loss: 0.0135
- Final Validation Loss: 0.1891
- Final Training Accuracy: 99.64%
- Final Validation Accuracy: 96.25%

#### LSTM Model:
- Final Training Loss: 0.0102
- Final Validation Loss: 0.6733
- Final Training Accuracy: 99.71%
- Final Validation Accuracy: 88.25%

### Analysis of Results:
- **BASIC Model**: The basic model provides decent performance, with high training and validation accuracy. The test accuracy is a bit lower, likely due to overfitting on the training data, though it still offers a competitive result. The simplicity of the model allows it to be trained quickly and efficiently.
  
- **CNN Model**: The CNN performs excellently on the training data with near-perfect accuracy (99.64%). However, the test accuracy slightly drops to 94.50%. This indicates that while the CNN is very good at learning patterns from the data, it struggles a bit with generalizing to unseen test data, possibly due to overfitting during training. This is reflected in the relatively higher validation loss compared to training loss.

- **LSTM Model**: Despite achieving very high training accuracy (99.71%), the LSTM performs poorly on validation data with only 88.25% accuracy. It could indicate that the model is overfitting to the training data, which results in a much higher validation loss. The large discrepancy between training and validation accuracy suggests that the model might be overfitting, particularly because the LSTM is not ideal for this type of feature-based data. As we know LSTMs are primarily designed for sequential data, where the order of the data is important.LSTMs can be used for classification tasks, especially when the data has a temporal or sequential structure. 

## Results Visualization

You can visualize the results and model performance using plots generated during training. These plots include:

- **Training vs. Validation Accuracy**: To compare how well each model performs on the training and validation sets.
- **Training vs. Validation Loss**: To show the convergence of the models and identify potential overfitting.
- **Confusion Matrix**: To visualize how well the model classifies different classes.

## Conclusion

- The **BASIC model** shows good general performance but lacks the complexity and power of the CNN and LSTM models.
- The **CNN model** shows excellent training accuracy but has slightly lower generalization performance due to overfitting.
- The **LSTM model** achieves high training accuracy but struggles with generalization, possibly due to overfitting or inadequate handling of sequential data.

In general, for this type of classification task, the **CNN** is the most effective model, while the **LSTM** might require further tuning or a different approach to handle the data effectively.

## How to Use
To run this project, clone the repository and execute `main.py` to train and evaluate the models. The user can choose to input their own features or let the system generate random features.

```bash
git clone https://github.com/wyattPol/ai_lab_final_project.git
cd ai_lab_final_project
python main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
