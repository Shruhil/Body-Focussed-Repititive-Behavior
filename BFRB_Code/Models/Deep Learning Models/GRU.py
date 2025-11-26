import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, GRU, Dense, Dropout, 
                                     BatchNormalization, Bidirectional,
                                     LayerNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 GRU MODEL TRAINING")
print("="*70)

# ==============================
# CONFIGURATION
# ==============================
ACTIVITIES = {
    'nail_biting': {
        'files': [
            {'imu_file': '/home/ankit/MPC/MPC_Project/denoised_signal_nail_biting_Person1.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_nail_biting_Person1.csv'},
            {'imu_file': '/home/ankit/MPC/MPC_Project/denoised_signal_nail_biting_Person2.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_nail_biting_Person2.csv'},
            {'imu_file': '/home/ankit/MPC/denoised_signal_nail_biting_Ananya.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_nail_biting_Ananya.csv'},
            {'imu_file': '/home/ankit/MPC/MPC_Project/denoised_signal_nail_biting.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_nail_biting.csv'},
        ],
        'label': 'nail_biting'
    },
    'beard_pulling': {
        'files': [
            {'imu_file': '/home/ankit/MPC/MPC_Project/beard_pulling_filtered_Person1.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_for_beard_pulling_Person1.csv'},
            {'imu_file': '/home/ankit/MPC/MPC_Project/beard_pulling_filtered_Person2.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_for_beard_pulling_Person2.csv'},
            {'imu_file': '/home/ankit/MPC/MPC_Project/beard_pulling_filtered.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_for_beard_pulling.csv'},
        ],
        'label': 'beard_pulling'
    },
    'face_itching': {
        'files': [
            {'imu_file': '/home/ankit/MPC/MPC_Project/denoised_signal_for_face_itching_Person1.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_for_face_itching_Person1.csv'},
            {'imu_file': '/home/ankit/MPC/MPC_Project/denoised_signal_for_face_itching_Person2.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_for_face_itching_Person2.csv'},
            {'imu_file': '/home/ankit/MPC/MPC_Project/denoised_signal_for_face_itching_Ananya.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_for_face_itching_Ananya.csv'},
        ],
        'label': 'face_itching'
    },
    'hair_pulling': {
        'files': [
            {'imu_file': '/home/ankit/MPC/MPC_Project/denoised_signal_for_hair_pulling_Person1.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_for_hair__pulling_Person1.csv'},
            {'imu_file': '/home/ankit/MPC/MPC_Project/denoised_signal_for_hair_pulling_Person2.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_for_hair__pulling_Person2.csv'},
            {'imu_file': '/home/ankit/MPC/MPC_Project/denoised_signal_for_hair_pulling_Ananya.csv',
             'annotation_file': '/home/ankit/MPC/MPC_Project/annotations_aligned_for_hair__pulling_Ananya.csv'},
        ],
        'label': 'hair_pulling'
    }
}

WINDOW_SIZE_SEC = 2.0
STEP_SIZE_SEC = 1.0
SAMPLING_RATE = 50
WINDOW_SIZE = int(WINDOW_SIZE_SEC * SAMPLING_RATE)
STEP_SIZE = int(STEP_SIZE_SEC * SAMPLING_RATE)
IMU_CHANNELS = ['AccelerationX', 'AccelerationY', 'AccelerationZ',
                'RotationX', 'RotationY', 'RotationZ']

# ==============================
# DATA LOADING FUNCTIONS
# ==============================
def augment_window(window):
    augmented = [window]
    augmented.append(window + np.random.normal(0, 0.01, window.shape))
    augmented.append(window * np.random.uniform(0.95, 1.05))
    augmented.append(np.roll(window, np.random.randint(-5, 5), axis=0))
    return augmented

def process_single_file(imu_file, annotation_file, activity_label, file_num):
    print(f"   📄 File {file_num}: {imu_file.split('/')[-1]}")
    try:
        imu_df = pd.read_csv(imu_file)
        ann_df = pd.read_csv(annotation_file)
        ann_df.columns = ann_df.columns.str.strip()
        
        imu_time = (imu_df['Timestamp'].values - imu_df['Timestamp'].iloc[0]) if 'Timestamp' in imu_df.columns else np.arange(len(imu_df)) / SAMPLING_RATE
        imu_data = imu_df[IMU_CHANNELS].values
        
        time_start_col = next((c for c in ann_df.columns if 'start' in c.lower()), None)
        time_end_col = next((c for c in ann_df.columns if 'end' in c.lower()), None)
        activity_col = next((c for c in ann_df.columns if 'activity' in c.lower() or 'label' in c.lower()), None)
        
        if not all([time_start_col, time_end_col, activity_col]):
            return [], []
        
        ann_df[activity_col] = ann_df[activity_col].astype(str).str.strip().str.lower()
        
        windows, labels = [], []
        activity_keywords = {'nail_biting': ['nail'], 'beard_pulling': ['beard'], 
                           'face_itching': ['face', 'itch'], 'hair_pulling': ['hair']}
        
        for _, row in ann_df.iterrows():
            label = row[activity_col]
            if label in ["0", "nan", ""] or pd.isna(label):
                continue
            
            start_idx = np.searchsorted(imu_time, row[time_start_col])
            end_idx = np.searchsorted(imu_time, row[time_end_col])
            
            if end_idx - start_idx < WINDOW_SIZE:
                continue
            
            for i in range(start_idx, end_idx - WINDOW_SIZE + 1, STEP_SIZE):
                window = imu_data[i:i+WINDOW_SIZE]
                final_label = label
                
                for keyword in activity_keywords.get(activity_label, []):
                    if keyword in label:
                        final_label = activity_label
                        break
                
                if final_label not in ['idle', 'false alarm', 'false_alarm']:
                    for aug_window in augment_window(window):
                        windows.append(aug_window)
                        labels.append(final_label)
                else:
                    windows.append(window)
                    labels.append(final_label)
        
        print(f"      ✓ {len(windows)} windows")
        return windows, labels
    except Exception as e:
        print(f"      ✗ Error: {e}")
        return [], []

# ==============================
# LOAD DATA
# ==============================
print("\n📁 Loading data...")
all_windows, all_labels = [], []

for activity_name, activity_info in ACTIVITIES.items():
    print(f"\n🔄 {activity_name.upper()}")
    for file_num, file_info in enumerate(activity_info['files'], 1):
        windows, labels = process_single_file(file_info['imu_file'], file_info['annotation_file'], 
                                             activity_info['label'], file_num)
        all_windows.extend(windows)
        all_labels.extend(labels)

X_all = np.array(all_windows, dtype=np.float32)
y_all = np.array(all_labels)

# Normalize data
print("\n🔧 Normalizing data...")
X_normalized = np.zeros_like(X_all)
for i in range(len(X_all)):
    for j in range(X_all.shape[2]):
        channel = X_all[i, :, j]
        X_normalized[i, :, j] = (channel - channel.mean()) / (channel.std() + 1e-8)

# Label encoding
label_to_int = {label: idx for idx, label in enumerate(sorted(np.unique(y_all)))}
int_to_label = {idx: label for label, idx in label_to_int.items()}
y_mapped = np.array([label_to_int[y] for y in y_all])

# Filter rare classes
MIN_SAMPLES = 10
unique, counts = np.unique(y_mapped, return_counts=True)
classes_to_keep = unique[counts >= MIN_SAMPLES]
mask = np.isin(y_mapped, classes_to_keep)
X_normalized = X_normalized[mask]
y_mapped = np.array([list(classes_to_keep).index(y) for y in y_mapped[mask]])
int_to_label = {i: int_to_label[old_idx] for i, old_idx in enumerate(classes_to_keep)}

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_mapped), y=y_mapped)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
)

num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"\n✅ Train: {len(X_train)}, Test: {len(X_test)}, Classes: {num_classes}")

# ==============================
# BUILD GRU MODEL
# ==============================
print("\n🏗️  Building GRU model...")

inputs = Input(shape=X_train.shape[1:])

# First GRU block
x = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2))(inputs)
x = LayerNormalization()(x)
x = Dropout(0.3)(x)

# Second GRU block
x = Bidirectional(GRU(64, return_sequences=True, recurrent_dropout=0.2))(x)
x = LayerNormalization()(x)
x = Dropout(0.3)(x)

# Third GRU block
x = Bidirectional(GRU(32, return_sequences=False, recurrent_dropout=0.2))(x)
x = LayerNormalization()(x)
x = Dropout(0.3)(x)

# Dense layers
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile with additional metrics
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 
             Precision(name='precision'),
             Recall(name='recall')]
)

model.summary()

# ==============================
# TRAIN MODEL
# ==============================
print("\n🚀 Training GRU...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_gru.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=150,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# ==============================
# EVALUATE MODEL
# ==============================
print("\n📊 Evaluating model...")

# Get predictions
predictions = model.predict(X_test, verbose=0)
pred_classes = np.argmax(predictions, axis=1)

# Calculate all metrics using sklearn
accuracy = accuracy_score(y_test, pred_classes)
f1_weighted = f1_score(y_test, pred_classes, average='weighted')
f1_macro = f1_score(y_test, pred_classes, average='macro')
precision_weighted = precision_score(y_test, pred_classes, average='weighted', zero_division=0)
precision_macro = precision_score(y_test, pred_classes, average='macro', zero_division=0)
recall_weighted = recall_score(y_test, pred_classes, average='weighted', zero_division=0)
recall_macro = recall_score(y_test, pred_classes, average='macro', zero_division=0)

# Get loss from model evaluation
loss, acc_keras, prec_keras, rec_keras = model.evaluate(X_test, y_test_cat, verbose=0)

# Print comprehensive results
print(f"\n{'='*70}")
print(f"✅ GRU MODEL PERFORMANCE METRICS")
print(f"{'='*70}")
print(f"\n📈 Overall Metrics:")
print(f"   Accuracy          : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Loss              : {loss:.4f}")
print(f"\n🎯 Weighted Metrics (accounting for class imbalance):")
print(f"   F1-Score (Weighted)    : {f1_weighted:.4f}")
print(f"   Precision (Weighted)   : {precision_weighted:.4f}")
print(f"   Recall (Weighted)      : {recall_weighted:.4f}")
print(f"\n🎯 Macro Metrics (equal weight per class):")
print(f"   F1-Score (Macro)       : {f1_macro:.4f}")
print(f"   Precision (Macro)      : {precision_macro:.4f}")
print(f"   Recall (Macro)         : {recall_macro:.4f}")
print(f"\n📊 Improvement:")
print(f"   From baseline (4.17%)  : {(accuracy - 0.0417)*100:+.2f}%")
print(f"{'='*70}")

# Detailed classification report
target_names = [int_to_label[i] for i in range(num_classes)]
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, pred_classes, target_names=target_names, zero_division=0))

# Per-class metrics
print("\n📊 Per-Class Performance:")
print(f"{'Class':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 88)

for i in range(num_classes):
    mask = y_test == i
    if np.sum(mask) > 0:
        class_acc = np.mean(pred_classes[mask] == y_test[mask])
        class_prec = precision_score(y_test, pred_classes, labels=[i], average=None, zero_division=0)[0]
        class_rec = recall_score(y_test, pred_classes, labels=[i], average=None, zero_division=0)[0]
        class_f1 = f1_score(y_test, pred_classes, labels=[i], average=None, zero_division=0)[0]
        support = np.sum(mask)
        
        print(f"{int_to_label[i]:<20} {class_acc:<12.4f} {class_prec:<12.4f} {class_rec:<12.4f} {class_f1:<12.4f} {support:<10}")

# ==============================
# VISUALIZATIONS
# ==============================
print("\n📊 Generating visualizations...")

# 1. Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'})
plt.title(f'GRU Confusion Matrix\nAccuracy: {accuracy:.4f} | F1-Score: {f1_weighted:.4f}', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.tight_layout()
plt.savefig('gru_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: gru_confusion_matrix.png")
plt.show()

# 2. Training History (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2, color='blue')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, linestyle='--', color='orange')
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2, color='blue')
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2, linestyle='--', color='orange')
axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2, color='green')
axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2, linestyle='--', color='red')
axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2, color='purple')
axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2, linestyle='--', color='brown')
axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('GRU Training History - All Metrics', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('gru_training_history.png', dpi=300, bbox_inches='tight')
print("✅ Saved: gru_training_history.png")
plt.show()

# 3. Metrics Comparison Bar Chart
fig, ax = plt.subplots(figsize=(12, 8))
metrics_names = ['Accuracy', 'F1-Score\n(Weighted)', 'Precision\n(Weighted)', 'Recall\n(Weighted)', 
                 'F1-Score\n(Macro)', 'Precision\n(Macro)', 'Recall\n(Macro)']
metrics_values = [accuracy, f1_weighted, precision_weighted, recall_weighted, 
                  f1_macro, precision_macro, recall_macro]
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#34495e']

bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('GRU Model - Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.xticks(rotation=0, ha='center')
plt.tight_layout()
plt.savefig('gru_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: gru_metrics_comparison.png")
plt.show()

# ==============================
# SAVE MODEL & METADATA
# ==============================
model.save('gru_final.h5')
print("✅ Saved: gru_final.h5")

metadata = {
    'model_type': 'GRU',
    'accuracy': float(accuracy),
    'loss': float(loss),
    'f1_score_weighted': float(f1_weighted),
    'f1_score_macro': float(f1_macro),
    'precision_weighted': float(precision_weighted),
    'precision_macro': float(precision_macro),
    'recall_weighted': float(recall_weighted),
    'recall_macro': float(recall_macro),
    'improvement_from_baseline': float((accuracy - 0.0417) * 100),
    'num_classes': num_classes,
    'label_mapping': {'int_to_label': int_to_label, 'label_to_int': label_to_int},
    'window_size': WINDOW_SIZE,
    'sampling_rate': SAMPLING_RATE,
    'per_class_metrics': {}
}

# Add per-class metrics to metadata
for i in range(num_classes):
    mask = y_test == i
    if np.sum(mask) > 0:
        metadata['per_class_metrics'][int_to_label[i]] = {
            'accuracy': float(np.mean(pred_classes[mask] == y_test[mask])),
            'precision': float(precision_score(y_test, pred_classes, labels=[i], average=None, zero_division=0)[0]),
            'recall': float(recall_score(y_test, pred_classes, labels=[i], average=None, zero_division=0)[0]),
            'f1_score': float(f1_score(y_test, pred_classes, labels=[i], average=None, zero_division=0)[0]),
            'support': int(np.sum(mask))
        }

with open('gru_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Saved: gru_metadata.json")

print("\n" + "="*70)
print("🎉 GRU TRAINING COMPLETE!")
print("="*70)