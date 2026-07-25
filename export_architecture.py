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

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

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
# 2+3. Custom IEEE-style architecture diagram
#      (pure matplotlib — no Graphviz needed)
# -----------------------------------------------
def draw_architecture_diagram(out_dir, dpi=300):
    """
    Draw a compact, publication-quality block diagram of the hybrid model.
    Landscape layout: 14 x 8 inches — suitable for IEEE double-column figures.
    """
    FW, FH = 14, 8
    fig, ax = plt.subplots(figsize=(FW, FH))
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── colour palette ──────────────────────────────────────────────
    C_IN     = '#DDEEFF'   # input
    C_CNN    = '#D4EDDA'   # CNN branch  (green)
    C_VIT    = '#FFF3CD'   # ViT branch  (amber)
    C_SE     = '#FFE0B2'   # SE-Block    (orange)
    C_FUSE   = '#E8EAF6'   # fusion      (indigo)
    C_HEAD   = '#F3E5F5'   # head        (purple)
    C_OUT    = '#E0F2F1'   # output      (teal)
    BORDER   = '#333333'
    ARROW    = dict(arrowstyle='->', color=BORDER, lw=1.6)

    def box(cx, cy, w, h, title, sub='', fc='#F5F5F5',
            fs=8.5, bold=False, ls='-'):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle='round,pad=0.07',
                              facecolor=fc, edgecolor=BORDER,
                              linewidth=1.3, linestyle=ls)
        ax.add_patch(rect)
        dy = 0.13 if sub else 0
        ax.text(cx, cy + dy, title, ha='center', va='center',
                fontsize=fs, fontweight='bold' if bold else 'normal',
                color='#111111')
        if sub:
            ax.text(cx, cy - dy, sub, ha='center', va='center',
                    fontsize=6.8, color='#555555', style='italic')

    def arr(x1, y1, x2, y2, rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(**ARROW,
                                   connectionstyle=f'arc3,rad={rad}'))

    def dim_label(cx, cy, txt):
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=6.5,
                color='#444444',
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec='none', alpha=0.85))

    # ══════════════════════════════════════════════════════════════════
    # TITLE
    ax.text(FW/2, 7.72,
            'EfficientNetB4 + Vision Transformer Hybrid Model Architecture',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # ══════════════════════════════════════════════════════════════════
    # INPUT  (centred top)
    box(7, 7.1, 3.4, 0.6, 'Input Image', '380 x 380 x 3',
        fc=C_IN, fs=9.5, bold=True)

    # split arrows
    arr(5.5, 6.8, 3.0, 6.25, rad=0.15)     # -> CNN
    arr(8.5, 6.8, 11.0, 6.25, rad=-0.15)   # -> ViT

    # ══════════════════════════════════════════════════════════════════
    # CNN BRANCH  (left column  x=2.4)
    CX = 2.4
    ax.text(CX, 6.1, 'CNN Branch', ha='center', fontsize=9,
            fontweight='bold', color='#1B5E20')

    box(CX, 5.55, 4.0, 0.75,
        'EfficientNetB4 Backbone', 'Frozen | ImageNet pre-trained',
        fc=C_CNN, fs=8.5)
    arr(CX, 5.17, CX, 4.65); dim_label(CX, 4.92, '12x12x1792')

    box(CX, 4.3, 4.0, 0.6,
        'Reshape  +  Linear Proj.', '144 tokens x 64-dim',
        fc=C_CNN, fs=8.5)
    arr(CX, 4.0, CX, 3.45); dim_label(CX, 3.72, 'CNN tokens (144x64)')

    # ── CNN token flows into fusion ──
    arr(CX, 3.15, 5.2, 2.12, rad=0.25)

    # ══════════════════════════════════════════════════════════════════
    # ViT BRANCH  (right column  x=11.6)
    VX = 11.6
    ax.text(VX, 6.1, 'ViT Branch', ha='center', fontsize=9,
            fontweight='bold', color='#7B1FA2')

    box(VX, 5.55, 4.0, 0.75,
        'Patch Extraction  +  Patch Encoder',
        'Patch 16x16  |  529 patches  |  64-dim projection',
        fc=C_VIT, fs=8.5)
    arr(VX, 5.17, VX, 4.52)

    # Transformer stack (dashed border = repeated block)
    rect_t = FancyBboxPatch((9.6, 3.7), 4.0, 0.75,
                            boxstyle='round,pad=0.07',
                            facecolor=C_VIT, edgecolor=BORDER,
                            linewidth=1.3, linestyle='--')
    ax.add_patch(rect_t)
    ax.text(VX, 4.21, 'Transformer Encoder Block  x 6',
            ha='center', va='center', fontsize=8.5, fontweight='bold')
    ax.text(VX, 3.92,
            'LayerNorm  →  MHA (4 heads, 64-dim)  →  Add  |  LayerNorm  →  FFN(GELU)  →  Add',
            ha='center', va='center', fontsize=6.8, color='#333333')

    arr(VX, 3.7, VX, 3.18)

    box(VX, 2.88, 4.0, 0.55,
        'SE-Block (Channel Attention)', '529 x 64-dim',
        fc=C_SE, fs=8.5)
    arr(VX, 2.60, VX, 2.13); dim_label(VX, 2.37, 'ViT tokens (529x64)')

    # ── ViT token flows into fusion ──
    arr(VX, 1.85, 8.8, 1.62, rad=-0.2)

    # ══════════════════════════════════════════════════════════════════
    # CROSS-ATTENTION FUSION  (centre)
    fx, fy, fw, fh = 7.0, 1.7, 5.6, 1.05
    frect = FancyBboxPatch((fx - fw/2, fy - fh/2), fw, fh,
                           boxstyle='round,pad=0.1',
                           facecolor=C_FUSE, edgecolor=BORDER, linewidth=1.8)
    ax.add_patch(frect)
    ax.text(fx, fy + 0.28,
            'Bidirectional Cross-Attention Fusion',
            ha='center', va='center', fontsize=9.5, fontweight='bold',
            color='#1A237E')
    ax.text(fx, fy + 0.04,
            'ViT queries CNN context  |  CNN queries ViT context  (4 heads, 64-dim)',
            ha='center', va='center', fontsize=7.2, color='#333333')
    ax.text(fx, fy - 0.20,
            'Add & LayerNorm  →  GlobalAvgPool  →  Concatenate  →  Dropout',
            ha='center', va='center', fontsize=7.2, color='#333333')
    ax.text(fx, fy - 0.40,
            'Output: 128-dim fused representation',
            ha='center', va='center', fontsize=7.0, color='#555555',
            style='italic')

    arr(fx, fy - fh/2, fx, 0.72)
    dim_label(fx, 0.95, '128-dim')

    # ══════════════════════════════════════════════════════════════════
    # CLASSIFIER HEAD  (centre bottom)
    box(7.0, 0.42, 5.6, 0.55,
        'Dense(256, ReLU)  +  Dropout(0.3)  ->  Dense(3, Softmax)',
        'Total: 18.5M params  |  Trainable: 874K  |  Loss: Categorical Cross-Entropy',
        fc=C_HEAD, fs=8.5, bold=False)

    # ── Class output boxes ──
    classes = ['Healthy', 'Red Rot', 'Smut']
    clr     = ['#C8E6C9', '#FFCDD2', '#FFF9C4']
    for i, (cls, col) in enumerate(zip(classes, clr)):
        cx2 = 5.3 + i * 1.7
        rect2 = FancyBboxPatch((cx2 - 0.72, -0.36), 1.44, 0.42,
                               boxstyle='round,pad=0.06',
                               facecolor=col, edgecolor=BORDER, linewidth=1.0)
        ax.add_patch(rect2)
        ax.text(cx2, -0.15, cls, ha='center', va='center',
                fontsize=8.5, fontweight='bold')
        arr(7.0, 0.14, cx2, -0.14)

    plt.tight_layout(pad=0.1)

    # Save
    png_path = out_dir / 'hybrid_model_architecture.png'
    pdf_path = out_dir / 'hybrid_model_architecture.pdf'
    fig.savefig(str(png_path), dpi=dpi, bbox_inches='tight',
                facecolor='white')
    fig.savefig(str(pdf_path), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('[2] Architecture PNG  ->  ' + str(png_path.resolve()))
    print('[3] Architecture PDF  ->  ' + str(pdf_path.resolve()))


draw_architecture_diagram(OUT_DIR, dpi=300)


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
