# Designing and Training Keras Models for Image Recognition

This document summarizes key design choices when building convolutional neural networks (CNNs) with Keras for image-classification tasks. Each section includes detailed explanations, code examples, and tips for experimentation.

---

## 1. Convolutional Layers

Convolutional layers extract local features (edges, textures) by applying learnable filters over the input.

### 1.1 Number of Filters

* **Early layers:** 32–64 filters. Capture basic shapes.
* **Deeper layers:** 128–256 filters. Capture complex patterns.
* **Trade-off:** More filters → better representation, but ↑compute and parameters.

```python
# two conv layers with increasing filter count
tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')
tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')
```

### 1.2 Kernel Size

* **3×3 (recommended):** small receptive field, stackable, fewer parameters.
* **5×5 or 7×7:** larger receptive field, but often replaced by two 3×3 layers:

  * Two 3×3 layers have a combined receptive field of 5×5, with fewer parameters.

```python
# equivalent receptive field:
x = Conv2D(64, (3,3), activation='relu')(x)
x = Conv2D(64, (3,3), activation='relu')(x)
```

### 1.3 Strides & Padding

* **padding='same':** keeps spatial dimensions, useful before pooling.
* **stride=2:** downsamples directly (similar to pooling) but less common than explicit pooling.

```python
x = Conv2D(64, (3,3), strides=2, padding='same', activation='relu')(x)
```

---

## 2. Pooling Layers

Pooling downsamples feature maps, reducing spatial size and computational cost.

* **MaxPooling2D (2×2):** selects strongest activation in each window.
* **AveragePooling2D (2×2):** averages activations; can smooth noise.
* **Placement:** typically after one or two conv layers.

```python
x = MaxPooling2D(pool_size=(2,2))(x)
```

---

## 3. Depth & Stacking Strategy

* **Shallow nets:** 2–3 conv+pool stacks for simple tasks or small images.
* **Deep nets:** 5+ stacks (e.g., VGG, ResNet) for complex datasets.
* **Residual blocks (ResNet):** skip connections help gradients flow in very deep architectures.

```python
# simple residual block
def res_block(x, filters):
    shortcut = x
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = add([x, shortcut])
    return Activation('relu')(x)
```

---

## 4. Dense Layers & Classifier Head

After feature extraction, flatten or pool, then connect to fully connected layers.

* **Flatten → Dense:** standard approach.
* **GlobalAveragePooling2D → Dense:** reduces overfitting, fewer parameters.

```python
# Flatten head
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
```

```python
# Global average pooling head
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
```

---

## 5. Regularization: Dropout & Batch Normalization

* **Dropout:** randomly zeroes a fraction of activations.

  * Typical rates: 0.2–0.5.
  * Place after dense layers or conv blocks.

```python
x = Dropout(0.3)(x)
```

* **BatchNormalization:** normalizes layer inputs.

  * Speeds up convergence, reduces sensitivity to initialization.

```python
x = Conv2D(64, (3,3), padding='same', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
```

---

## 6. Optimizer & Learning Rate

* **Adam:** good default, requires minimal tuning.
* **SGD + momentum:** often yields better final accuracy with learning-rate schedules.

### 6.1 Learning-Rate Scheduling

```python
from tensorflow.keras.optimizers.schedules import ExponentialDecay
lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.5,
    staircase=True
)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
```

---

## 7. Data Augmentation

Generate realistic variations to reduce overfitting:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)
model.fit(
    aug.flow(x_train, y_train, batch_size=32),
    epochs=EPOCHS,
    validation_data=(x_test, y_test)
)
```

*Note:* Avoid flips on directional traffic signs.

---

## 8. Batch Size & Epochs

* **Batch size:** 16–64. Larger batches smooth gradients; may need higher LR.
* **Epochs:** monitor validation loss; use early stopping.

```python
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(..., epochs=50, callbacks=[es])
```

---

## 9. Full Example Model

```python
from tensorflow.keras import Input, Model
inputs = Input(shape=(30,30,3))

# Block 1
x = Conv2D(32,(3,3),padding='same',use_bias=False)(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)

# Block 2
x = Conv2D(64,(3,3),padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Dropout(0.25)(x)

# Head
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(43, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 10. Callbacks & Monitoring

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    TensorBoard(log_dir='./logs', histogram_freq=1)
]
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    callbacks=callbacks
)
```

---

## 11. Tips for Experimentation

* Change **one** hyperparameter at a time. Keep a lab notebook!
* Track results with **TensorBoard** or **Weights & Biases**.
* Try both **Adam** and **SGD+momentum**.
* Evaluate **different depths** (e.g., 3-block vs. 5-block architectures).
* Experiment with **dropout rates** and presence/absence of batch norm.

---

By systematically iterating through these components, you’ll hone in on an architecture that balances accuracy, training time, and parameter count for your traffic-sign recognition task.
