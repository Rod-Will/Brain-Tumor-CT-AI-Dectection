```markdown
# Brain Tumor Classification Using CNN and Pre-trained Models  

This repository provides an implementation of brain tumor classification using convolutional neural networks (CNNs) and transfer learning with pre-trained models like VGG16, ResNet50, and DenseNet121. The dataset includes CT scan images labeled as "Healthy" or "Tumor."  

## Features  

- **Custom CNN Model**: Built from scratch using TensorFlow and Keras.  
- **Transfer Learning**: Utilizes pre-trained VGG16, ResNet50, and DenseNet121 models for improved accuracy and efficiency.  
- **Performance Metrics**: Tracks accuracy, AUC, F1-score, and generates visualizations like ROC curves, accuracy plots, and confusion matrices.  
- **Dataset Preparation**: Automatically splits the dataset into training, validation, and test sets, with an option to use a 20% subset.  
- **Comparative Analysis**: Compares model performance and identifies the best-performing model.  

## Dataset  

The dataset contains CT scan images labeled into two categories:  
1. **Healthy**  
2. **Tumor**

![Sample_Images](https://github.com/user-attachments/assets/b36bab59-5156-4a30-af24-3dccd9e9ae41)

The dataset is automatically preprocessed to create a 10% subset with an 80-20-10 split for training, validation, and testing.  

## Requirements  

Ensure the following dependencies are installed:  

- Python 3.8+  
- TensorFlow 2.0+  
- NumPy  
- Matplotlib  
- Pandas  
- Seaborn  
- scikit-learn  

Install the dependencies with:  

```bash
pip install -r requirements.txt
```  

## Usage  

### Step 1: Clone the Repository  

```bash
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification
```  

### Step 2: Prepare Dataset  

Place the dataset folder at `./CT/Brain Tumor CT scan Images` with subfolders `Healthy` and `Tumor`.  

### Step 3: Train and Evaluate Models  

Run the script to train and evaluate models:  

```bash
python train_models.py
```  

### Outputs  

- **Models**: Saved in `outcome04/models/` as `.keras` files.  
- **Plots**: Stored in `outcome04/plots/` (ROC curve, confusion matrix, accuracy plots, and comparative metrics).  
- **Results**: Classification reports and metrics summary saved in `outcome04/results/`.  

### Step 4: Analyze Results  

The best-performing model is displayed, and a bar chart comparing the metrics for all models is generated.  

## Results  

- **Performance Metrics**:  
  - **Accuracy**: Reported for each model.  
  - **AUC**: Area under the ROC curve.  
  - **F1-Score**: Balance between precision and recall.  

- **Comparative Metrics**:
![Comparative_Metrics](https://github.com/user-attachments/assets/e9bbfcb2-470e-420d-a1ae-5e3f0fed0a60)

## License  

This project is licensed under the [Creative Commons Zero v1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).  

## Acknowledgments  

- The dataset was sourced from [Kaggle Brain Tumor Dataset](https://www.kaggle.com/).  
- Pre-trained model architectures (VGG16, ResNet50, DenseNet121) are from the TensorFlow Keras Applications.  
- Visualizations are powered by Matplotlib and Seaborn.  

## References  

- [TensorFlow Documentation](https://www.tensorflow.org/)  
- [Keras Pre-trained Models](https://keras.io/api/applications/)  
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)  

## Contact  

For questions, feedback, or contributions, contact:  

- **Name**: Rod Will  
- **Email**: rhudwill@gmail.com  
- **GitHub**: [https://github.com/Rod-Will]

---

Thank you for exploring this project! Contributions and feedback are always welcome.  
```  
