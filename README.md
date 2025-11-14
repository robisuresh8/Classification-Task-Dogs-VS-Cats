# Dogs vs. Cats Image Classification

This project is a deep learning implementation of the classic "Dogs vs. Cats" image classification task. The goal is to build and train a **Convolutional Neural Network (CNN)** that can accurately distinguish between images of dogs and cats.

The entire process, from loading the data to building the model and training it, is contained within the `Classification_Task_Dogs_VS_Cats.ipynb` Jupyter Notebook.

## Project Workflow

1.  **Data Preprocessing:** Images are loaded from their respective folders (`training_set` and `test_set`).
2.  **Image Augmentation:** Keras's `ImageDataGenerator` is used to apply transformations (like shearing, zooming, and flipping) to the training images. This helps prevent overfitting and creates a more robust model.
3.  **Model Building:** A sequential **CNN** is built using Keras, consisting of:
    * `Conv2D` layers for feature extraction.
    * `MaxPooling2D` layers to reduce dimensionality.
    * A `Flatten` layer to convert the 2D feature maps into a 1D vector.
    * `Dense` (fully connected) layers to perform the classification.
    * A final `Dense` layer with a **sigmoid** activation function for binary (dog or cat) output.
4.  **Model Training:** The model is compiled with an `adam` optimizer and `binary_crossentropy` loss function, then trained on the dataset.
5.  **Evaluation:** The model's performance is evaluated on the test set.

## Technology Stack

* **Python**
* **Jupyter Notebook**
* **TensorFlow**
* **Keras** (specifically `Sequential` models, `Conv2D`, `MaxPooling2D`, `Dense`, and `ImageDataGenerator`)
* **NumPy**

## How to Run This Project

### 1. Clone the Repository

```bash
git clone [https://github.com/robisuresh8/Classification-Task-Dogs-VS-Cats.git](https://github.com/robisuresh8/Classification-Task-Dogs-VS-Cats.git)
cd Classification-Task-Dogs-VS-Cats
```

### 2. Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

You will need to install TensorFlow, Keras (which is part of TF), and Jupyter.

```bash
pip install jupyterlab tensorflow numpy
```

### 4. Download the Dataset

This repository **does not include the dataset**. You will need to download the "Dogs vs. Cats" dataset, typically from a source like Kaggle.

* **Link to the Kaggle competition:** [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

After downloading, you must structure the data in the following way inside the project's root directory:
```
.
├── dataset/
│   ├── training_set/
│   │   ├── cats/
│   │   │   ├── cat.1.jpg
│   │   │   └── ...
│   │   └── dogs/
│   │       ├── dog.1.jpg
│   │       └── ...
│   └── test_set/
│       ├── cats/
│       │   ├── cat.4001.jpg
│       │   └── ...
│       └── dogs/
│           ├── dog.4001.jpg
│           └── ...
└── Classification_Task_Dogs_VS_Cats.ipynb
```
**Note:** You may need to adjust the paths inside the `Classification_Task_Dogs_VS_Cats.ipynb` notebook to match the directory you create (e.g., `dataset/training_set`).

### 5. Run the Jupyter Notebook

Start Jupyter Lab to open and run the notebook.

```bash
jupyter lab
```

From the browser, open `Classification_Task_Dogs_VS_Cats.ipynb` and run the cells sequentially to train the model.
