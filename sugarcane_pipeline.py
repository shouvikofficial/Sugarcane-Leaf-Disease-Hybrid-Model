import os
import random
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, GlobalAveragePooling1D, LayerNormalization, MultiHeadAttention, Concatenate, Input, Add, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, matthews_corrcoef, cohen_kappa_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

import optuna
import kagglehub
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ==========================================
# SET RANDOM SEEDS FOR REPRODUCIBILITY
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================
# FIGURE OUTPUT CONFIG
# ==========================================
FIGURES_DIR = Path("outputs") / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SHOW_FIGURES = os.environ.get("SHOW_FIGURES", "0") == "1"
RUN_EXPLAINABILITY = os.environ.get("RUN_EXPLAINABILITY", "1") == "1"
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"
PRINT_MODEL_SUMMARY_TERMINAL = os.environ.get("PRINT_MODEL_SUMMARY_TERMINAL", "0") == "1"
FIGURE_DPI = int(os.environ.get("FIGURE_DPI", "600"))
FIGURE_FORMATS = tuple(
    fmt.strip().lower() for fmt in os.environ.get("FIGURE_FORMATS", "png,pdf").split(",") if fmt.strip()
)
EXPORTED_FIGURES = []

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": FIGURE_DPI,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
})

print(f"Figure output directory: {FIGURES_DIR.resolve()}")
print(
    "SHOW_FIGURES="
    f"{SHOW_FIGURES} | RUN_EXPLAINABILITY={RUN_EXPLAINABILITY} | "
    f"FORCE_RETRAIN={FORCE_RETRAIN} | PRINT_MODEL_SUMMARY_TERMINAL={PRINT_MODEL_SUMMARY_TERMINAL} | "
    f"FIGURE_DPI={FIGURE_DPI} | FIGURE_FORMATS={FIGURE_FORMATS}"
)

def finalize_plot(filename=None):
    """Save current figure in all requested formats and optionally show it."""
    if filename:
        base_name = Path(filename).stem
        for fmt in FIGURE_FORMATS:
            out_path = FIGURES_DIR / f"{base_name}.{fmt}"
            plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
            EXPORTED_FIGURES.append(out_path)
            print(f"Saved figure: {out_path.resolve()}")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()

# ==========================================
# 1. DOWNLOAD AND PREPARE DATASET
# ==========================================
print("Downloading dataset via Kagglehub...")
dataset_path = kagglehub.dataset_download("ingjuanquevedo/sugarcane-smut-dataset")
print(f"Dataset saved at: {dataset_path}")

# Gather image paths and labels
image_paths = []
labels = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, file))
            labels.append(os.path.basename(root))

df = pd.DataFrame({'filepath': image_paths, 'label': labels})
print(f"Total images before deduplication: {len(df)}")

# Compute MD5 hash for each image and remove exact duplicates globally
print("\nComputing file hashes for global deduplication...")

def compute_md5(filepath):
    if not os.path.exists(filepath):
        return None
    return hashlib.md5(Path(filepath).read_bytes()).hexdigest()

df["file_hash"] = df["filepath"].apply(compute_md5)
missing_files = df["file_hash"].isna().sum()
if missing_files > 0:
    print(f"Warning: {missing_files} files were missing and will be dropped.")
    df = df.dropna(subset=["file_hash"]).copy()

duplicate_mask = df.duplicated(subset=["file_hash"], keep="first")
duplicate_count = int(duplicate_mask.sum())
if duplicate_count > 0:
    duplicate_by_class = df.loc[duplicate_mask, "label"].value_counts().to_dict()
    print(f"Found {duplicate_count} duplicate images. Removing globally before split.")
    print(f"Removed duplicates by class: {duplicate_by_class}")
    df = df.loc[~duplicate_mask].copy()
else:
    print("No exact duplicate images found for global deduplication.")

print(f"Total images after deduplication: {len(df)}")

# Stratified split on deduplicated dataset: 70% Train, 15% Val, 15% Test
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ==========================================
# 1.5 DATA LEAKAGE CHECK (MD5 HASH)
# ==========================================
print("\nChecking for duplicate images between train and test sets...")
train_hashes = set(train_df["file_hash"].tolist())
test_hashes = set(test_df["file_hash"].tolist())
val_hashes = set(val_df["file_hash"].tolist())

train_test_overlap = train_hashes.intersection(test_hashes)
train_val_overlap = train_hashes.intersection(val_hashes)
val_test_overlap = val_hashes.intersection(test_hashes)

duplicates = len(train_test_overlap) + len(train_val_overlap) + len(val_test_overlap)

if duplicates:
    print(
        "⚠️ Leakage check failed! "
        f"train-test: {len(train_test_overlap)}, "
        f"train-val: {len(train_val_overlap)}, "
        f"val-test: {len(val_test_overlap)}"
    )
else:
    print("✅ No identical image files found across train/val/test splits! (No Data Leakage)")

# ==========================================
# 2. DATA GENERATORS & AUGMENTATION
# ==========================================
IMG_SIZE = 380
BATCH_SIZE = 16

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='filepath', y_col='label', target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)
val_generator = test_datagen.flow_from_dataframe(
    val_df, x_col='filepath', y_col='label', target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)
test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col='filepath', y_col='label', target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# Calculate Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
weight_dict = dict(enumerate(class_weights))
class_names = list(train_generator.class_indices.keys())


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = Dense(projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.squeeze = Dense(channels // ratio, activation="relu")
        self.excite = Dense(channels, activation="sigmoid")

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=1, keepdims=True)
        x = self.squeeze(x)
        x = self.excite(x)
        return tf.keras.layers.Multiply()([inputs, x])


def transformer_encoder(inputs, num_heads=4, key_dim=64, ff_dim=256, dropout_rate=0.2):
    x1 = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(x1, x1)
    x2 = Add()([attention_output, inputs])

    x3 = LayerNormalization(epsilon=1e-6)(x2)
    x3 = Dense(ff_dim, activation="gelu")(x3)
    x3 = Dense(inputs.shape[-1])(x3)

    return Add()([x3, x2])


def vit_branch(inputs, patch_size=16, projection_dim=64, transformer_layers=6, num_heads=4, dropout_rate=0.1, return_tokens=False):
    num_patches = (IMG_SIZE // patch_size) ** 2
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    x = encoded_patches
    for _ in range(transformer_layers):
        x = transformer_encoder(
            x,
            num_heads=num_heads,
            key_dim=projection_dim,
            ff_dim=projection_dim * 2,
            dropout_rate=dropout_rate,
        )
        x = SEBlock(channels=projection_dim, ratio=8)(x)

    x = LayerNormalization(epsilon=1e-6)(x)
    if return_tokens:
        return x

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    return x


def cross_attention_fusion(cnn_feature_map, vit_tokens, projection_dim=64, num_heads=4, dropout_rate=0.1):
    """Bidirectional cross-attention fusion between CNN and ViT token streams."""
    cnn_tokens = tf.keras.layers.Reshape((-1, cnn_feature_map.shape[-1]))(cnn_feature_map)
    cnn_tokens = Dense(projection_dim)(cnn_tokens)
    vit_tokens = Dense(projection_dim)(vit_tokens)

    # ViT queries CNN context
    vit_from_cnn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        dropout=dropout_rate,
    )(query=vit_tokens, key=cnn_tokens, value=cnn_tokens)
    vit_fused = Add()([vit_tokens, vit_from_cnn])
    vit_fused = LayerNormalization(epsilon=1e-6)(vit_fused)

    # CNN queries ViT context
    cnn_from_vit = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=projection_dim,
        dropout=dropout_rate,
    )(query=cnn_tokens, key=vit_tokens, value=vit_tokens)
    cnn_fused = Add()([cnn_tokens, cnn_from_vit])
    cnn_fused = LayerNormalization(epsilon=1e-6)(cnn_fused)

    vit_pooled = GlobalAveragePooling1D()(vit_fused)
    cnn_pooled = GlobalAveragePooling1D()(cnn_fused)

    fused = Concatenate()([cnn_pooled, vit_pooled])
    fused = Dropout(dropout_rate)(fused)
    return fused


def build_hybrid_model(dense_units, dropout_rate, learning_rate, optimizer_name, num_classes):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    backbone = EfficientNetB4(weights="imagenet", include_top=False, input_tensor=inputs)
    backbone.trainable = False

    cnn_feature_map = backbone.output
    vit_tokens = vit_branch(
        inputs,
        patch_size=16,
        projection_dim=64,
        transformer_layers=6,
        num_heads=4,
        dropout_rate=dropout_rate,
        return_tokens=True,
    )

    combined = cross_attention_fusion(
        cnn_feature_map,
        vit_tokens,
        projection_dim=64,
        num_heads=4,
        dropout_rate=dropout_rate,
    )
    combined = Dense(dense_units, activation='relu')(combined)
    combined = Dropout(dropout_rate)(combined)
    outputs = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate) if optimizer_name == "Adam" else RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model, backbone

# ==========================================
# 3. OPTUNA HYPERPARAMETER TUNING
# ==========================================
FINAL_MODEL_PATH = Path("Sugarcane_Hybrid_Final.keras")
BEST_WEIGHTS_PATH = Path("best_efficientnet_b4.keras")

if FINAL_MODEL_PATH.exists() and not FORCE_RETRAIN:
    print(f"\nFound existing trained model: {FINAL_MODEL_PATH.resolve()}")
    print("Skipping Optuna/training. Set FORCE_RETRAIN=1 to train again.")
    model = tf.keras.models.load_model(
        FINAL_MODEL_PATH,
        custom_objects={
            "Patches": Patches,
            "PatchEncoder": PatchEncoder,
            "SEBlock": SEBlock,
        },
        compile=False,
    )
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    try:
        base_model = model.get_layer("efficientnetb4")
    except ValueError:
        base_model = None
        print("Warning: EfficientNet backbone layer not found by name in loaded model.")

    # Keep plotting/evaluation pipeline compatible when training is skipped
    history = type("HistoryObj", (), {"history": {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}})()
    history_finetune = type("HistoryObj", (), {"history": {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}})()
    lr_schedule = []
    TRAINING_RAN = False
else:
    def objective(trial):
        tf.keras.backend.clear_session()

        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
        dense_units = trial.suggest_categorical("dense_units", [128, 256, 512])

        model, _ = build_hybrid_model(
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            num_classes=len(class_names),
        )

        history = model.fit(train_generator, validation_data=val_generator, epochs=5, class_weight=weight_dict, verbose=1)
        return max(history.history["val_accuracy"])

    print("\nStarting Optuna Study...")
    study = optuna.create_study(
        study_name="sugarcane_efficientnet_study",
        storage="sqlite:///sugarcane_optuna_study.db",  # Saves locally in your VS Code folder
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )
    TARGET_TOTAL_TRIALS = 25
    finished_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    completed_trials = sum(1 for t in study.trials if t.state in finished_states)
    remaining_trials = max(0, TARGET_TOTAL_TRIALS - completed_trials)

    print(f"Existing trials in DB: {completed_trials}")
    print(f"Remaining trials to run: {remaining_trials}")

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=True)
    else:
        print(f"Target already reached ({TARGET_TOTAL_TRIALS} trials). Skipping optimization.")
    print("Best Params:", study.best_params)

    # ==========================================
    # 4. BUILD & TRAIN THE BEST MODEL
    # ==========================================
    tf.keras.backend.clear_session()
    best = study.best_params
    model, base_model = build_hybrid_model(
        dense_units=best["dense_units"],
        dropout_rate=best["dropout_rate"],
        learning_rate=best["learning_rate"],
        optimizer_name=best["optimizer"],
        num_classes=len(class_names),
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
        ModelCheckpoint(str(BEST_WEIGHTS_PATH), save_best_only=True)
    ]

    # Learning rate tracking callback
    lr_schedule = []

    class LRLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            lr_schedule.append(lr)

    callbacks.append(LRLogger())

    print("\nTraining best model for 20 epochs...")
    history = model.fit(
        train_generator, validation_data=val_generator, epochs=20,
        class_weight=weight_dict, callbacks=callbacks
    )

    # ==========================================
    # 5. FINE-TUNING
    # ==========================================
    print("\nUnfreezing top layers for fine-tuning...")
    for layer in base_model.layers[-50:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    history_finetune = model.fit(
        train_generator, validation_data=val_generator, epochs=10,
        class_weight=weight_dict, callbacks=callbacks
    )
    TRAINING_RAN = True

# ==========================================
# 6. EVALUATION & VISUALIZATION
# ==========================================
if FORCE_RETRAIN or not FINAL_MODEL_PATH.exists():
    if BEST_WEIGHTS_PATH.exists():
        model.load_weights(str(BEST_WEIGHTS_PATH))
    model.save(str(FINAL_MODEL_PATH))
    print(f"Final model saved as: {FINAL_MODEL_PATH}")
else:
    print(f"Using existing saved model for evaluation: {FINAL_MODEL_PATH}")
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

pred = model.predict(test_generator)
pred_class = np.argmax(pred, axis=1)
true_class = test_generator.classes

print("\nClassification Report:")
report_dict = classification_report(true_class, pred_class, target_names=class_names, output_dict=True)
print(classification_report(true_class, pred_class, target_names=class_names))
print("\nGenerating and saving evaluation figures...")

# Confusion Matrix
cm = confusion_matrix(true_class, pred_class)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
finalize_plot("confusion_matrix.png")

# ==========================================
# 6.5 ADVANCED EVALUATION METRICS
# ==========================================
# ROC-AUC Curve
y_test_binarized = label_binarize(true_class, classes=np.arange(len(class_names)))
n_classes = y_test_binarized.shape[1]

plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve for {class_names[i]} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Multi-class Curve')
plt.legend(loc="lower right")
finalize_plot("roc_auc_curve.png")

# Dynamic F1-Score Chart
f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
plt.figure(figsize=(6, 4))
plt.bar(class_names, f1_scores, color=['#4C72B0', '#55A868', '#C44E52'])
plt.ylim(0.0, 1.05)
plt.title('F1-Score per Class')
plt.ylabel('F1-Score')
finalize_plot("f1_scores.png")

# ==========================================
# 6.6 TRAINING CURVES & ADDITIONAL METRICS
# ==========================================
if TRAINING_RAN:
    # Combine training histories
    all_accuracy = history.history['accuracy'] + history_finetune.history['accuracy']
    all_val_accuracy = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
    all_loss = history.history['loss'] + history_finetune.history['loss']
    all_val_loss = history.history['val_loss'] + history_finetune.history['val_loss']
    epochs_initial = len(history.history['accuracy'])

    # Plot Training & Validation Loss/Accuracy (both stages)
    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs_initial), history.history['accuracy'], label='Train Accuracy (Initial)', linewidth=2)
    plt.plot(range(epochs_initial), history.history['val_accuracy'], label='Val Accuracy (Initial)', linewidth=2)
    plt.plot(range(epochs_initial, epochs_initial + len(history_finetune.history['accuracy'])), 
             history_finetune.history['accuracy'], label='Train Accuracy (Fine-tune)', linewidth=2, linestyle='--')
    plt.plot(range(epochs_initial, epochs_initial + len(history_finetune.history['val_accuracy'])), 
             history_finetune.history['val_accuracy'], label='Val Accuracy (Fine-tune)', linewidth=2, linestyle='--')
    plt.axvline(x=epochs_initial, color='red', linestyle=':', alpha=0.5, label='Fine-tuning Start')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy (Initial + Fine-tune)')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs_initial), history.history['loss'], label='Train Loss (Initial)', linewidth=2)
    plt.plot(range(epochs_initial), history.history['val_loss'], label='Val Loss (Initial)', linewidth=2)
    plt.plot(range(epochs_initial, epochs_initial + len(history_finetune.history['loss'])), 
             history_finetune.history['loss'], label='Train Loss (Fine-tune)', linewidth=2, linestyle='--')
    plt.plot(range(epochs_initial, epochs_initial + len(history_finetune.history['val_loss'])), 
             history_finetune.history['val_loss'], label='Val Loss (Fine-tune)', linewidth=2, linestyle='--')
    plt.axvline(x=epochs_initial, color='red', linestyle=':', alpha=0.5, label='Fine-tuning Start')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss (Initial + Fine-tune)')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    finalize_plot("training_curves.png")

    # Plot Learning Rate Schedule
    if lr_schedule:
        plt.figure(figsize=(10, 4))
        plt.plot(lr_schedule, linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule (ReduceLROnPlateau)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        finalize_plot("learning_rate_schedule.png")
        print(f"Learning rate tracked over {len(lr_schedule)} epochs")
else:
    print("Training curves skipped because existing trained model was loaded.")

# ==========================================
# 6.7 ADDITIONAL STATISTICAL METRICS
# ==========================================

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(true_class, pred_class)
print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")

# Cohen's Kappa Score
kappa = cohen_kappa_score(true_class, pred_class)
print(f"Cohen's Kappa Score: {kappa:.4f}")

# Macro, Weighted, Micro Averages
print("\n" + "="*60)
print("AGGREGATED METRICS")
print("="*60)
print(f"Accuracy (Micro): {report_dict['accuracy']:.4f}")
print(f"Precision (Macro): {report_dict['macro avg']['precision']:.4f}")
print(f"Precision (Weighted): {report_dict['weighted avg']['precision']:.4f}")
print(f"Recall (Macro): {report_dict['macro avg']['recall']:.4f}")
print(f"Recall (Weighted): {report_dict['weighted avg']['recall']:.4f}")
print(f"F1-Score (Macro): {report_dict['macro avg']['f1-score']:.4f}")
print(f"F1-Score (Weighted): {report_dict['weighted avg']['f1-score']:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Kappa: {kappa:.4f}")

# ==========================================
# 6.8 PRECISION-RECALL CURVES
# ==========================================

plt.figure(figsize=(12, 4))

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], pred[:, i])
    ap = average_precision_score(y_test_binarized[:, i], pred[:, i])
    plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AP={ap:.4f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
finalize_plot("precision_recall_curves.png")

# ==========================================
# 6.9 PER-CLASS METRICS TABLE
# ==========================================

metrics_data = []
for cls_name in class_names:
    metrics_data.append({
        'Class': cls_name,
        'Precision': f"{report_dict[cls_name]['precision']:.4f}",
        'Recall': f"{report_dict[cls_name]['recall']:.4f}",
        'F1-Score': f"{report_dict[cls_name]['f1-score']:.4f}",
        'Support': int(report_dict[cls_name]['support'])
    })

metrics_df = pd.DataFrame(metrics_data)
print("\n" + "="*60)
print("PER-CLASS PERFORMANCE SUMMARY")
print("="*60)
print(metrics_df.to_string(index=False))

# ==========================================
# 6.10 MODEL ARCHITECTURE SUMMARY
# ==========================================

print("\n" + "="*60)
print("MODEL ARCHITECTURE SUMMARY")
print("="*60)
if PRINT_MODEL_SUMMARY_TERMINAL:
    model.summary()
else:
    print("Model summary terminal print is disabled. See model_summary.txt")

total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Non-trainable Parameters: {non_trainable_params:,}")
print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (assuming float32)")

# Save model summary to file
with open('model_summary.txt', 'w', encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write(f"\nTotal Parameters: {total_params:,}\n")
    f.write(f"Trainable Parameters: {trainable_params:,}\n")
    f.write(f"Non-trainable Parameters: {non_trainable_params:,}\n")
    f.write(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (assuming float32)\n")
print("Model summary saved to model_summary.txt")

# ==========================================
# 7. EXPLAINABLE AI (LIME & GRAD-CAM) FOR HYBRID MODEL
# ==========================================

# --- GRAD-CAM FOR EFFICIENTNET BRANCH ---
def make_gradcam_heatmap_hybrid(img_array, hybrid_model, last_conv_layer_name, pred_index=None):
    """Grad-CAM for the EfficientNetB4 branch of the hybrid model."""
    grad_model = tf.keras.models.Model(
        inputs=hybrid_model.inputs,
        outputs=[hybrid_model.get_layer(last_conv_layer_name).output, hybrid_model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam_hybrid(img_array, heatmap, true_label, pred_label, confidence, alpha=0.5, save_path=None):
    """Display Grad-CAM result with original image and heatmap overlay."""
    img = np.uint8(img_array[0])
    
    heatmap_resized = tf.image.resize(heatmap[np.newaxis, :, :, np.newaxis], (IMG_SIZE, IMG_SIZE))
    heatmap_resized = heatmap_resized[0, :, :, 0].numpy()
    
    heatmap_colored = np.uint8(255 * heatmap_resized)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_colored]
    
    superimposed_img = np.uint8(alpha * (jet_heatmap * 255) + (1 - alpha) * img)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Original Image\nTrue: {true_label}", fontweight='bold')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM (EfficientNet Branch)\nPred: {pred_label} ({confidence*100:.1f}%)", fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    finalize_plot(save_path)


# --- LIME FOR HYBRID MODEL ---
def predict_fn_for_lime(images):
    """Wrapper for LIME to predict on the hybrid model."""
    images = preprocess_input(images.copy())
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)
    return model.predict(images, verbose=0)


print("\n" + "="*50)
print("EXPLAINABILITY ANALYSIS: LIME & GRAD-CAM")
print("="*50)

if RUN_EXPLAINABILITY:
    # Run LIME and Grad-CAM on 3 random test images
    for i in range(3):
        random_idx = random.randint(0, len(test_df) - 1)
        sample_img_path = test_df['filepath'].iloc[random_idx]
        true_label = test_df['label'].iloc[random_idx]
        
        img = tf.keras.utils.load_img(sample_img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = tf.keras.utils.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        preds = model.predict(preprocess_input(img_array_expanded.copy()), verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_label = class_names[pred_idx]
        confidence = np.max(preds[0])
        
        print(f"\n--- Image {i+1}: {true_label} (Pred: {pred_label}, {confidence*100:.1f}%) ---")
        
        # Grad-CAM on EfficientNetB4 branch - Auto-detect last Conv2D layer
        print("Computing Grad-CAM on EfficientNetB4 branch...")
        last_conv_layer = None
        if base_model is not None:
            for layer in reversed(base_model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break
        else:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break
        if last_conv_layer is None:
            print("Warning: Could not find Conv2D layer for Grad-CAM")
        else:
            heatmap = make_gradcam_heatmap_hybrid(preprocess_input(img_array_expanded.copy()), model, last_conv_layer)
            display_gradcam_hybrid(img_array_expanded, heatmap, true_label, pred_label, confidence, save_path=f"gradcam_image_{i+1}.png")
        
        # LIME on full hybrid model
        print("Computing LIME explanation (this may take a moment)...")
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            (img_array / 255.0).astype(np.float32),  # LIME expects 0-1 range
            predict_fn_for_lime,
            top_labels=3,
            hide_color=0,
            num_samples=300
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        
        plt.figure(figsize=(10, 6))
        plt.imshow(mark_boundaries(temp, mask))
        plt.title(f"LIME Explanation (Full Hybrid Model)\nTrue: {true_label} | Pred: {pred_label}", fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        finalize_plot(f"lime_explanation_{i+1}.png")

    print("\n" + "="*50)
    print("Explanation analysis complete!")
    print("="*50)
else:
    print("Explainability stage skipped. Set RUN_EXPLAINABILITY=1 to enable it.")

if EXPORTED_FIGURES:
    print("\n" + "="*50)
    print("EXPORTED FIGURES")
    print("="*50)
    for fig_path in sorted(set(EXPORTED_FIGURES)):
        print(fig_path.resolve())

 