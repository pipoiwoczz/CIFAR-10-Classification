# 📚 CIFAR-10 Image Classification (MLP Benchmark Across Libraries)

This project presents an implementation and comparison of Multi-Layer Perceptron (MLP) models on the CIFAR-10 dataset using four machine learning libraries:

- [x] Scikit-learn
- [x] TensorFlow / Keras
- [x] PyTorch
- [x] JAX (with Flax)

The objective is to evaluate each framework based on performance (training time, GPU memory usage), ease of implementation, and training results — all using a **consistent architecture** and methodology.

---

## 🎯 Project Objective

Explore and benchmark popular ML libraries by implementing the same MLP model in:

- **Scikit-learn** for simplicity
- **TensorFlow** for high-level abstraction
- **PyTorch** for low-level flexibility
- **JAX** for performance and functional programming style

This allows for a fair comparison between ease of use, training efficiency, and model behavior.

---

## 🧠 Model Architecture

All implementations use the **same MLP structure**:

- **Input Layer**: 3072 neurons (flattened 32×32×3 images)
- **Hidden Layers**:
  - Layer 1: 1024 units (ReLU)
  - Layer 2: 512 units (ReLU)
  - Layer 3: 256 units (ReLU)
- **Output Layer**: 10 neurons (Softmax)
- **Dropout**: 0.3 applied between hidden layers
- **Loss Function**: Cross-Entropy
- **Optimizer**: Adam, learning rate = 0.0001
- **Early Stopping**: patience = 5 (monitors validation accuracy)

---

## 🗂 Project Structure

```bash
CIFAR-10-Classification/
├── Source.ipynb              # Notebook with implementations for all 4 frameworks
├── Report.pdf                # Final analysis and self-evaluation report
├── Requirements.pdf          # Project specifications from course instructor
└── README.md                 # You are here
```

---

## 🏁 Running the Project

1. Clone the repository:

```bash
git clone https://github.com/pipoiwoczz/CIFAR-10-Classification.git
cd CIFAR-10-Classification
```

2. Open the notebook:

```bash
jupyter notebook Source.ipynb
```

3. Make sure you install dependencies according to the framework you're testing:

```bash
pip install scikit-learn tensorflow torch torchvision jax jaxlib flax matplotlib
```

---

## 🧪 Framework Comparison

| Framework     | Training Time (s) | GPU Usage (MB) | Notes                                                  |
|---------------|------------------:|----------------:|---------------------------------------------------------|
| **Scikit-learn** | 816              | 0               | Easy but slow, no GPU support, limited deep learning     |
| **TensorFlow**   | 54               | 1157            | High-level API, fast, good community                     |
| **PyTorch**      | 57               | 1314            | Flexible, explicit training loop, high performance       |
| **JAX**          | 34               | 10              | Fastest, lowest memory, hardest to implement manually    |

---

## 📝 Subjective Evaluation

| Framework     | Pros                                                                 | Cons                                        |
|---------------|----------------------------------------------------------------------|---------------------------------------------|
| Scikit-learn  | Very easy to implement, beginner-friendly                            | Slow, no GPU, limited customization          |
| TensorFlow    | Easy to use, good community, built-in support for dropout/tracking   | Higher memory usage                          |
| PyTorch       | Powerful, widely used, transparent training process                  | Requires more manual setup                   |
| JAX           | High-performance, very efficient on GPU                              | Steep learning curve, limited documentation  |

---

## 📊 Results Summary

- **Best Training Speed**: JAX  
- **Best Memory Efficiency**: JAX  
- **Best Balance for Beginners**: TensorFlow  
- **Most Transparent for Learning**: PyTorch  
- **Simplest Interface**: Scikit-learn  

---

## 📌 References

- CIFAR-10 Dataset: [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Keras Docs](https://www.tensorflow.org/api_docs/python/tf/keras)
- [PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)
- [JAX Docs](https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html)

---

## 👨‍🎓 Author
- **Lê Ngọc Anh Khoa**  
  - Student ID: 22127196  
  - University of Science – HCMUS  
  - Course: Introduction to Machine Learning (CSC14005)
- **Github**: [Github](https://github.com/pipoiwoczz)
- **Portfolio**: [Portfolio](https://pipoiwoczz.vercel.app/)

This project is for academic use under the terms of the course. Reuse is welcome with attribution.
