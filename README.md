
#Food Classification Using Pre-Trained Models with Data Augmentation and Learning Rate Adjustment#

Project Overview
This project focuses on implementing a Convolutional Neural Network (CNN) to classify images of food using a Kaggle dataset. I initially attempted to train a custom model but found its performance unsatisfactory. Instead, I used a pre-trained model (EfficientNetB7) to perform transfer learning. While the pre-trained model significantly improved performance, the model still faced high variance, indicating overfitting. Attempts to mitigate overfitting using techniques such as Dropout and learning rate adjustment were only partially successful.

Dataset
Source: Kaggle Food Classification Dataset
Type: Image classification task
Preprocessing: Images were normalized to [0, 1] and augmented for training.
Methods and Techniques
1. Pre-Trained Model
Used EfficientNetB7 as the base model for transfer learning.
The top layers of the pre-trained model were fine-tuned while freezing the initial layers to utilize pre-trained features.
2. Data Augmentation
To prevent overfitting and improve model generalization, I applied data augmentation techniques, including:

Random Rotation: 10%
Random Zoom: 10%
Random Horizontal Flip
Code Example:

python
data_augmentation = tf.keras.Sequential([
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomFlip("horizontal"),
])
3. Learning Rate Adjustment
I implemented ReduceLROnPlateau to dynamically reduce the learning rate when validation loss plateaued.

Factor: Reduce learning rate by 0.5
Patience: 3 epochs
Minimum Learning Rate: 
1
×
1
0
−
6
1×10 
−6
 
Code Example:

python
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)
4. Regularization and Dropout
Added L2 regularization to the Dense layer to penalize large weights.
Used Dropout with a rate of 0.6 to reduce overfitting.
Model Architecture:

python
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(11, activation='softmax')
])
Training Results
Training Accuracy: 99%
Validation Accuracy: 88%
Observation: The model showed clear signs of high variance (overfitting). Despite using Dropout and learning rate adjustment, the validation performance plateaued.
Challenges Faced
Overfitting:

Training accuracy reached 99%, but validation accuracy remained at 88%.
Techniques like Dropout and data augmentation provided limited improvement.
Custom Model Performance:

Training a custom CNN model from scratch yielded very poor results.
Leveraging pre-trained models (EfficientNetB7) drastically improved performance.
High Variance:

Despite efforts to reduce overfitting, the model's high variance persisted. Further improvements like additional regularization, larger datasets, or advanced architectures are needed.
Conclusion
Using EfficientNetB7 as a pre-trained model with data augmentation and learning rate adjustment successfully improved the model performance but did not completely resolve overfitting. Future work will focus on:

Exploring other pre-trained models (e.g., ResNet, Inception).
Adding more advanced regularization techniques.
Collecting or generating a larger dataset to improve generalization.
References
Deep Learning Specialization by Andrew Ng (Coursera).
YOLO: Real-Time Object Detection paper.
