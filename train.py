import tensorflow as tf
import tensorflow_model_optimization as tfmot
from pathlib import Path
import numpy as np
import time

start = time.time()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print(f"GPU detected: {gpus}")
  tf.config.experimental.set_memory_growth(gpus[0], True)
else:
  print("WARNING: No GPU detected, training on CPU")


def load_and_preprocess(image_path, label, img_size=96):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_image(img, channels=1, expand_animations=False)
  img = tf.image.resize(img, [img_size, img_size])
  img = (img / 127.5) - 1.0
  return img, label


def augment(img, label):
  img = tf.image.random_flip_left_right(img)
  return img, label


def create_dataset(data_dir='images', img_size=96, batch_size=16, validation_split=0.2, seed=42):
  with_people = list(Path(data_dir).glob('with_people/*.png'))
  without_people = list(Path(data_dir).glob('without_people/*.png'))

  paths = [str(p) for p in with_people + without_people]
  labels = [1] * len(with_people) + [0] * len(without_people)

  np.random.seed(seed)
  indices = np.random.permutation(len(paths))
  split_idx = int(len(paths) * (1 - validation_split))

  train_paths = [paths[i] for i in indices[:split_idx]]
  train_labels = [labels[i] for i in indices[:split_idx]]
  val_paths = [paths[i] for i in indices[split_idx:]]
  val_labels = [labels[i] for i in indices[split_idx:]]

  train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
  train_ds = train_ds.shuffle(2000, seed=seed).map(
      lambda x, y: load_and_preprocess(x, y, img_size),
      num_parallel_calls=tf.data.AUTOTUNE
  ).map(augment, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

  val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
  val_ds = val_ds.map(
      lambda x, y: load_and_preprocess(x, y, img_size),
      num_parallel_calls=tf.data.AUTOTUNE
  ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

  return train_ds, val_ds


def cnn_model(img_size=96):
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(img_size, img_size, 1)),

      tf.keras.layers.Conv2D(
          16, 3, strides=2, padding='same', activation='relu'),

      tf.keras.layers.Conv2D(
          32, 3, strides=2, padding='same', activation='relu'),

      tf.keras.layers.Conv2D(
          48, 3, strides=2, padding='same', activation='relu'),
      tf.keras.layers.Dropout(0.2),

      # Classification head
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dropout(0.3),
      tf.keras.layers.Dense(
          1,
          activation='sigmoid',
          kernel_regularizer=tf.keras.regularizers.l2(0.01)
      )
  ], name='simple_cnn_person_detector')

  return model


def representative_dataset_generator(val_ds):
  def representative_dataset():
    print("Generating representative dataset for quantization calibration...")
    count = 0
    for images, _ in val_ds.unbatch().batch(1).take(100):
      yield [tf.cast(images, tf.float32)]
      count += 1
    print(f"Generated {count} calibration samples")
  return representative_dataset


def validate_quantized_model(tflite_model, val_ds, num_samples=100):
  print("\nValidating quantized model accuracy...")
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  print(f"Input details: {input_details[0]}")
  print(f"Output details: {output_details[0]}")

  correct = 0
  total = 0

  for images, labels in val_ds.unbatch().batch(1).take(num_samples):
    # Convert to int8 input format
    input_scale = input_details[0]['quantization'][0]
    input_zero_point = input_details[0]['quantization'][1]

    quantized_input = images / input_scale + input_zero_point
    quantized_input = tf.cast(quantized_input, tf.int8)

    interpreter.set_tensor(input_details[0]['index'], quantized_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize output
    output_scale = output_details[0]['quantization'][0]
    output_zero_point = output_details[0]['quantization'][1]
    dequantized_output = (output.astype(np.float32) -
                          output_zero_point) * output_scale

    pred = 1 if dequantized_output[0][0] > 0.5 else 0
    correct += (pred == labels.numpy()[0])
    total += 1

  accuracy = correct / total
  print(
      f"\nQuantized model accuracy on {num_samples} validation samples: {accuracy:.4f}")
  return accuracy


print("="*60)
print("TFLite Micro Person Detection Training Pipeline")
print("="*60)

print("\nCreating datasets...")
train_ds, val_ds = create_dataset()

print("\nBuilding simple CNN...")
model = cnn_model()

print("\nModel summary:")
model.summary()

# Phase 1: Initial training
print("\n" + "="*60)
print("Phase 1: Initial training (up to 100 epochs with early stopping)")
print("="*60)

# Calculate total steps for cosine decay
steps_per_epoch = None
for _ in train_ds:
  steps_per_epoch = (steps_per_epoch or 0) + 1

total_steps = steps_per_epoch * 100

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=total_steps
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

print(f"\nPhase 1 Results:")
print(f"  Training accuracy: {history1.history['accuracy'][-1]:.4f}")
print(f"  Validation accuracy: {history1.history['val_accuracy'][-1]:.4f}")

# QaT
print("\n" + "="*60)
print("Phase 2: Quantization-aware training (50 epochs)")
print("="*60)

print("Applying quantization to model...")
q_aware_model = tfmot.quantization.keras.quantize_model(model)

q_aware_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
history2 = q_aware_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    verbose=1
)

print(f"\nPhase 2 Results:")
print(f"  Training accuracy: {history2.history['accuracy'][-1]:.4f}")
print(f"  Validation accuracy: {history2.history['val_accuracy'][-1]:.4f}")

# Tflite conversition
print("\n" + "="*60)
print("Converting to TFLite INT8")
print("="*60)

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_generator(val_ds)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.experimental_new_quantizer = True

print("Running conversion...")
tflite_model = converter.convert()

output_path = 'person_detection.tflite'
with open(output_path, 'wb') as f:
  f.write(tflite_model)

model_size_kb = len(tflite_model) / 1024
print(f"\nModel saved to {output_path}")
print(f"Model size: {model_size_kb:.1f} KB")

quantized_accuracy = validate_quantized_model(
    tflite_model, val_ds, num_samples=100)
accuracy_drop = history2.history['val_accuracy'][-1] - quantized_accuracy


print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(
    f"Phase 1 (Initial training) val accuracy:  {history1.history['val_accuracy'][-1]:.4f}")
print(
    f"Phase 2 (QAT) val accuracy:               {history2.history['val_accuracy'][-1]:.4f}")
print(f"INT8 quantized accuracy:                  {quantized_accuracy:.4f}")
print(f"Accuracy drop from QAT:                   {accuracy_drop:.4f}")
print(f"\nModel size: {model_size_kb:.1f} KB")
print(f"Output file: {output_path}")

if accuracy_drop > 0.05:
  print("\nWARNING: Large accuracy drop (>5%) from quantization")
  print("Consider:")
  print("  - More QAT epochs")
  print("  - Better representative dataset coverage")
else:
  print(f"\n✓ Model ready for TFLite Micro deployment")

print("\n" + "="*60)
end = time.time()
print(f"Elapsed: {end - start} seconds")