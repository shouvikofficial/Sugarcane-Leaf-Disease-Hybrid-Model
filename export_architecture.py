"""
export_architecture.py
======================
Standalone script to export the Sugarcane Hybrid Model architecture.

Outputs
-------
outputs/architecture/
  ├── hybrid_model_architecture.png   ← High-res layer diagram (PNG)
  ├── hybrid_model_architecture.pdf   ← Vector layer diagram (PDF)
  └── hybrid_model.keras              ← Netron-compatible model file

Usage
-----
    python export_architecture.py

Then open hybrid_model.keras in Netron: https://netron.app

Requirements (for PNG/PDF)
--------------------------
    pip install pydot
    # Also install Graphviz binaries: https://graphviz.org/download/
"""

import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import (
    Add, Concatenate, Dense, Dropout, Embedding,
    GlobalAveragePooling1D, Input, LayerNormalization, MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────
# Configuration (must match sugarcane_pipeline.py)
# ─────────────────────────────────────────────
IMG_SIZE    = 380
NUM_CLASSES = 3        # Adjust if your dataset has a different number of classes
DENSE_UNITS = 256
DROPOUT     = 0.3
LR          = 1e-4

OUT_DIR = Path("outputs") / "architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Custom Layers (identical to pipeline)
# ─────────────────────────────────────────────
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

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches   = num_patches
        self.projection_dim = projection_dim
        self.projection        = Dense(projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_patches":    self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return cfg


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio    = ratio
        self.squeeze  = Dense(channels // ratio, activation="relu")
        self.excite   = Dense(channels, activation="sigmoid")

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=1, keepdims=True)
        x = self.squeeze(x)
        x = self.excite(x)
        return tf.keras.layers.Multiply()([inputs, x])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "ratio": self.ratio})
        return cfg


# ─────────────────────────────────────────────
# Architecture Functions (identical to pipeline)
# ─────────────────────────────────────────────
def transformer_encoder(inputs, num_heads=4, key_dim=64, ff_dim=256, dropout_rate=0.2):
    x1   = LayerNormalization(epsilon=1e-6)(inputs)
    attn = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate
    )(x1, x1)
    x2 = Add()([attn, inputs])
    x3 = LayerNormalization(epsilon=1e-6)(x2)
    x3 = Dense(ff_dim, activation="gelu")(x3)
    x3 = Dense(inputs.shape[-1])(x3)
    return Add()([x3, x2])


def vit_branch(inputs, patch_size=16, projection_dim=64,
               transformer_layers=6, num_heads=4,
               dropout_rate=0.1, return_tokens=False):
    num_patches = (IMG_SIZE // patch_size) ** 2
    patches     = Patches(patch_size)(inputs)
    x           = PatchEncoder(num_patches, projection_dim)(patches)

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


def cross_attention_fusion(cnn_feature_map, vit_tokens,
                           projection_dim=64, num_heads=4, dropout_rate=0.1):
    """Bidirectional cross-attention fusion between CNN and ViT token streams."""
    cnn_tokens = tf.keras.layers.Reshape((-1, cnn_feature_map.shape[-1]))(cnn_feature_map)
    cnn_tokens = Dense(projection_dim)(cnn_tokens)
    vit_tokens = Dense(projection_dim)(vit_tokens)

    # ViT queries CNN context
    vit_from_cnn = MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate
    )(query=vit_tokens, key=cnn_tokens, value=cnn_tokens)
    vit_fused = Add()([vit_tokens, vit_from_cnn])
    vit_fused = LayerNormalization(epsilon=1e-6)(vit_fused)

    # CNN queries ViT context
    cnn_from_vit = MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate
    )(query=cnn_tokens, key=vit_tokens, value=vit_tokens)
    cnn_fused = Add()([cnn_tokens, cnn_from_vit])
    cnn_fused = LayerNormalization(epsilon=1e-6)(cnn_fused)

    vit_pooled = GlobalAveragePooling1D()(vit_fused)
    cnn_pooled = GlobalAveragePooling1D()(cnn_fused)

    fused = Concatenate()([cnn_pooled, vit_pooled])
    fused = Dropout(dropout_rate)(fused)
    return fused


def build_hybrid_model(dense_units, dropout_rate, learning_rate, num_classes):
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
    combined = Dense(dense_units, activation="relu")(combined)
    combined = Dropout(dropout_rate)(combined)
    outputs  = Dense(num_classes, activation="softmax")(combined)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────
# Build model (no training, no dataset needed)
# ─────────────────────────────────────────────
print("Building hybrid model ...")
model = build_hybrid_model(
    dense_units=DENSE_UNITS,
    dropout_rate=DROPOUT,
    learning_rate=LR,
    num_classes=NUM_CLASSES,
)
print(f"Model built.  Total params: {model.count_params():,}")
model.summary()

# -----------------------------------------------
# 1. Save .keras  ->  open this in Netron
# -----------------------------------------------
keras_path = OUT_DIR / "hybrid_model.keras"
model.save(str(keras_path))
print("\n[1] Netron model file  ->  " + str(keras_path.resolve()))
print("     Drag it onto  https://netron.app  OR open with the Netron desktop app.")
print("     In Netron: File > Export (PNG / SVG)")

# -----------------------------------------------
# 2. PNG via tf.keras.utils.plot_model
# -----------------------------------------------
png_path = OUT_DIR / "hybrid_model_architecture.png"
try:
    tf.keras.utils.plot_model(
        model,
        to_file=str(png_path),
        show_shapes=True,
        show_layer_names=True,
        show_dtype=False,
        expand_nested=False,   # set True to expand EfficientNet internals (very tall)
        dpi=200,
        rankdir="TB",
    )
    print("[2] Architecture PNG  ->  " + str(png_path.resolve()))
except Exception as e:
    print("[2] PNG export failed: " + str(e))
    print("     Fix: pip install pydot  +  install Graphviz from https://graphviz.org/download/")

# -----------------------------------------------
# 3. PDF via tf.keras.utils.plot_model
# -----------------------------------------------
pdf_path = OUT_DIR / "hybrid_model_architecture.pdf"
try:
    tf.keras.utils.plot_model(
        model,
        to_file=str(pdf_path),
        show_shapes=True,
        show_layer_names=True,
        show_dtype=False,
        expand_nested=False,
        dpi=200,
        rankdir="TB",
    )
    print("[3] Architecture PDF  ->  " + str(pdf_path.resolve()))
except Exception as e:
    print("[3] PDF export failed: " + str(e))

# ─────────────────────────────────────────────
# 4. (Optional) Auto-launch Netron in browser
# ─────────────────────────────────────────────
try:
    import netron
    print(f"\n[4] Launching Netron ...  (press Ctrl+C to stop)")
    netron.start(str(keras_path))
except ImportError:
    print("\n[4] To launch Netron interactively:")
    print("     pip install netron")
    print("     python -c \"import netron; netron.start('outputs/architecture/hybrid_model.keras')\"")

print("\nDone.")
