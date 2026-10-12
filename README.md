# Orange Fruit Disease Detection

An image-classification project using TensorFlow, EfficientNet-B0 and
Streamlit to classify citrus fruit images into five categories.

## Disease Classes

- Black Spot
- Canker
- Greening
- Healthy
- Melanose

## Features

- Image upload through a Streamlit web interface
- Orange-versus-non-orange input gate
- Five-class disease predictions and confidence scores
- Grad-CAM visualizations to help interpret predictions
- Training, fine-tuning and evaluation scripts

## Technologies

Python, TensorFlow/Keras, EfficientNet-B0, Streamlit, NumPy,
Pillow, Matplotlib and scikit-learn.

## Installation and Usage

Create and activate a Python virtual environment, then install dependencies:

    pip install -r requirements.txt

Start the original application:

    streamlit run app.py

The application normally opens at http://localhost:8501.

## Model Evaluation

The original classifier achieved 95.96% test accuracy (95/99 correct).
The augmented experimental classifier achieved 95.60% validation accuracy
(87/91 correct) and 95.96% test accuracy (95/99 correct).

The test set is small, so these results are preliminary. The augmented
model improved validation accuracy but did not improve overall test accuracy.

## Limitations

The orange gate has accepted an apple-like fruit and rejected some genuine
orange images during testing. Manual override is available in the app.
This is an educational prototype, not a validated agricultural diagnostic
system.

## Dataset

Large datasets, extracted images, local environments and generated reports
are excluded from Git. Training scripts require the relevant datasets in
their expected local folders.
