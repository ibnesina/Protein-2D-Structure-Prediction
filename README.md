# Protein 2D Structure Prediction using Deep Learning

This project implements a Convolutional Neural Network (CNN) for predicting protein secondary structure from amino acid sequences. The model predicts three secondary structure classes: Helix (H), Coil (C), and Beta-sheet (B) for each amino acid residue in a protein sequence.

## 🧬 Project Overview

Protein secondary structure prediction is a fundamental problem in computational biology and bioinformatics. This project demonstrates how deep learning techniques can be applied to predict the 2D structural elements of proteins based solely on their amino acid sequences.

### What is Protein Secondary Structure?
- **Helix (H)**: Alpha-helices, spiral structures stabilized by hydrogen bonds
- **Beta-sheet (B)**: Extended strands connected by hydrogen bonds
- **Coil (C)**: Irregular structures that don't fall into the above categories

## 📁 Project Structure

```
1907002_Code/
├── protein_2d_structure_prediction.ipynb    # Main Jupyter notebook
├── sequence.fasta                           # Input protein sequences in FASTA format
├── Protein Structures/                      # Secondary structure annotations (.ss2 files)
│   ├── AAB51171.1.ss2
│   ├── AAB51177.1.ss2
│   ├── XP_054179810.1.ss2
│   └── ... (19 total protein structures)
├── weights_frames/                          # Training visualization frames
│   ├── epoch_1.png
│   ├── epoch_2.png
│   └── ... (100 training epochs)
└── weights_evolution.gif                    # Animated visualization of weight evolution
```

## 🚀 Features

- **Deep Learning Model**: 1D Convolutional Neural Network for sequence analysis
- **Data Processing**: One-hot encoding of amino acid sequences
- **Training Visualization**: Real-time weight evolution tracking with GIF generation
- **Performance Metrics**: Comprehensive evaluation including confusion matrix and classification reports
- **Data Augmentation**: Automatic sequence padding and label mapping

## 🛠️ Technical Implementation

### Model Architecture
The CNN model consists of:
- **Input Layer**: One-hot encoded amino acid sequences
- **Conv1D Layer 1**: 64 filters, kernel size 3, ReLU activation
- **Dropout Layer**: 30% dropout for regularization
- **Conv1D Layer 2**: 128 filters, kernel size 3, ReLU activation
- **Dropout Layer**: 30% dropout for regularization
- **Output Layer**: 3 filters (H, C, B), softmax activation

### Data Processing Pipeline
1. **Sequence Loading**: FASTA files containing protein sequences
2. **Structure Annotation**: SS2 files with secondary structure labels
3. **Encoding**: One-hot encoding of 20 standard amino acids
4. **Padding**: Sequences padded to maximum length with coil labels
5. **Label Mapping**: Secondary structure labels converted to integers (H:0, C:1, B:2)

### Training Configuration
- **Epochs**: 100
- **Batch Size**: 8
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Validation Split**: 30% (15% validation, 15% test)

## 📊 Results

The model achieves:
- **Test Accuracy**: 92.91%
- **Per-residue Accuracy**: 81.77%
- **Class-wise Performance**:
  - Helix (H): 77% precision, 78% recall
  - Coil (C): 86% precision, 88% recall
  - Beta-sheet (B): 67% precision, 52% recall

## 🎯 Usage

### Prerequisites
```bash
pip install tensorflow numpy scikit-learn biopython matplotlib seaborn imageio
```

### Running the Project
1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook protein_2d_structure_prediction.ipynb
   ```

2. **Execute Cells Sequentially**:
   - Cell 0: Import dependencies
   - Cell 1: Data loading and encoding functions
   - Cell 2: Load and preprocess data
   - Cell 3: Train/validation/test split
   - Cell 4: Build and compile model
   - Cell 5: Train the model
   - Cell 6: Evaluate performance
   - Cell 7: Plot training history
   - Cell 8: Generate confusion matrix and metrics

### Input Data Format
- **FASTA Files**: Standard FASTA format with protein sequences
- **SS2 Files**: Secondary structure annotation files with format:
  ```
  # Position  AA  SS
  1           M    H
  2           A    H
  3           V    H
  ```

## 🔬 Scientific Context

This project addresses the protein structure prediction problem, which is:
- **Computationally Challenging**: NP-hard problem in computational biology
- **Biologically Important**: Understanding protein structure aids drug discovery
- **Research Active**: Active area with CASP competitions and ongoing research

## 📈 Visualization Features

### Weight Evolution Tracking
The model includes a custom callback that:
- Captures Conv1D layer weights after each epoch
- Generates heatmap visualizations
- Creates an animated GIF showing weight evolution over training

### Training Metrics
- Loss curves for training and validation
- Accuracy progression over epochs
- Confusion matrix for final predictions

## 🤝 Contributing

This project demonstrates:
- Deep learning applications in bioinformatics
- Protein sequence analysis techniques
- CNN architectures for sequential data
- Training visualization and monitoring

## 📚 References

- **BioPython**: For FASTA file parsing
- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Visualization libraries

## 🔍 Future Enhancements

Potential improvements could include:
- **Attention Mechanisms**: For better sequence understanding
- **Transfer Learning**: Pre-trained protein language models
- **Ensemble Methods**: Combining multiple model predictions
- **Data Augmentation**: Synthetic sequence generation
- **Hyperparameter Tuning**: Automated optimization

## 📄 License

This project is for educational and research purposes. Please ensure proper attribution when using the code or methodology.

---

**Note**: This project demonstrates the application of deep learning to a fundamental problem in computational biology. The results show promising performance in predicting protein secondary structure from amino acid sequences alone.
