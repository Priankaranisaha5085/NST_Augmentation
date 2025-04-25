# NST_Augmentation
This project focuses on enhancing Breast Ultrasound (BUS) image datasets using Neural Style Transfer (NST) integrated with Explainable AI (XAI) techniques. The aim is to improve the quality, diversity, and interpretability of stylized augmented images. We employ Parallel Computing using the Horovod framework to accelerate the augmentation process.

To evaluate the effectiveness of our augmentation strategy, we classify both the original and the augmented datasets using a fine-tuned ResNet50 model.
## 🚀 Features
📸 Advanced Image Augmentation: Leverages Neural Style Transfer (NST) to enhance Breast Ultrasound (BUS) image diversity and quality, addressing overfitting in deep learning models.

🎨 Hybrid NST Architecture: Combines Demystifying NST (DNST) and mr2NST models with a pre-trained ResNet50 to design a novel and effective style loss function.

🧠 Explainable AI with LRP: Integrates Layer-wise Relevance Propagation (LRP) to highlight important features in content images, enhancing transparency and model interpretability.

⚡ Parallelized Training with Horovod: Utilizes Horovod for distributed deep learning, scaling across 8 GPUs for efficient NST training.

📊 Performance Evaluation: Assesses augmentation quality by training a fine-tuned ResNet50 classifier, showing a significant accuracy boost (from ~56% to 99.70%).

![Architecture Overview](images/13.png)


## 📂 Repository Structure
```
NST_Augmentation/
│
├── data/                   # Dataset directory
├── src/                    # Source code for NST and augmentation
│   ├── models/             # Pre-trained models for NST
│   ├── augmentation/       # Augmentation strategies and implementations
│   ├── utils/              # Helper functions and utilities
│
│
├── requirements.txt        # List of technologies and tools
├── README.md               # Project overview (this file)
├── CONTRIBUTING.md         # Contribution guidelines
└── LICENSE                 # License for the project
```

## 🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Priankaranisaha5085/NST_Augmentation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd NST_Augmentation
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Prepare your dataset by placing it in the `data/` directory.
2. Configure your augmentation and NST settings in the `src/` directory.
3. Run the main script to generate stylized images:
   ```bash
   python src/NST_Augmentation.py
   ```

## 📊 Examples and Results
Here are some examples of augmented and stylized outputs generated using this project:

### Original Image
![Original](examples/original.jpg)

### Stylized Image (After NST + Augmentation)
![Stylized](examples/stylized.jpg)

## 📜 License
This project is licensed under the [MIT License](LICENSE)


