"""
=============================================================
ctc_loss.py  —  CTC Loss Layer and Loss Function
=============================================================

WHY DO WE NEED CTC LOSS? — THE BIG PICTURE
--------------------------------------------
This is the most important concept in the entire OCR project.
Read this carefully.

PROBLEM WITH NORMAL LOSS FUNCTIONS:
Normal classification loss (Cross-Entropy) requires you to know
EXACTLY which character is at EACH position in the image.
For example, for the image of "cat":
  - Position 1 → 'c'
  - Position 2 → 'a'
  - Position 3 → 't'

But in handwriting, we DON'T know exactly where each character
starts and ends in the image. "c" might span pixels 0-40,
"a" from 41-80, "t" from 81-120 — but these boundaries are
not labeled in the dataset.

CTC (Connectionist Temporal Classification) SOLUTION:
CTC Loss solves this by NOT requiring position alignment.
You give it:
  - The model's output probability sequence (128 time steps)
  - The target label sequence ("cat" = [3, 1, 20])

CTC then considers ALL possible alignments that could produce "cat"
and trains the model to maximize the sum of those probabilities.

For example, all these model outputs decode to "cat":
  c c a a t        (2 c's, 2 a's, 1 t)
  c a a t t        (1 c, 2 a's, 2 t's)
  c [B] a [B] t    ([B] = blank token used as separator)
  c c [B] a t      etc.

CTC automatically handles:
  ✓ Variable text length
  ✓ Unknown character positions
  ✓ Repeated characters (uses blank to separate "aa" → "a[B]a" → "aa")

BLANK TOKEN:
CTC uses a special "blank" symbol (index 0) that means "nothing here"
or "same character continues". The blank helps separate identical
consecutive characters.
=============================================================
"""

import tensorflow as tf
from tensorflow import keras


class CTCLayer(keras.layers.Layer):
    """
    Custom Keras layer that computes CTC Loss during training.

    WHY A CUSTOM LAYER INSTEAD OF A LOSS FUNCTION?
    ------------------------------------------------
    The CTC Loss function needs FOUR inputs:
      1. Model predictions (logits)
      2. Target label sequence
      3. Length of prediction sequence (image_width // 4)
      4. Length of label sequence

    Standard Keras loss functions only receive 2 inputs (y_true, y_pred).
    By embedding the loss computation inside a custom Layer, we can
    pass all 4 inputs through the model's input dict and compute the
    loss inside the layer's call() method.

    The layer returns the loss value and adds it to the model's
    total loss via self.add_loss(). This integrates seamlessly with
    model.fit() without any custom training loops.
    """

    def __init__(self, name: str = "ctc_loss_layer", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        """
        Compute the CTC Loss and add it to the model's loss.

        HOW CTC LOSS WORKS MATHEMATICALLY:
        -----------------------------------
        CTC Loss = -log P(label | predictions)

        Where P(label | predictions) is the sum of probabilities of
        ALL possible alignments that decode to the target label.
        The model is trained to maximize this probability (minimize -log).

        Parameters
        ----------
        inputs : tuple of 4 tensors
            y_true       : tf.Tensor  shape (batch, MAX_LABEL_LEN)  int32
                           Padded integer-encoded label sequences.
                           Padding values (-1) are masked during loss computation.

            y_pred       : tf.Tensor  shape (batch, time_steps, num_classes)  float32
                           Raw logits from the final Dense layer of the CRNN.
                           time_steps = IMG_WIDTH // 4 (e.g. 512//4 = 128)
                           num_classes = len(ALPHABET) + 1 (e.g. 81)

            input_length : tf.Tensor  shape (batch,)  int32
                           Number of time steps for each image in the batch.
                           All values = time_steps (same for fixed-width images).

            label_length : tf.Tensor  shape (batch,)  int32
                           Actual length of each label (before padding).
                           Example: "Hi" → label_length = 2

        Returns
        -------
        tf.Tensor  — y_pred (passed through unchanged)
                     The loss is registered via self.add_loss().
        """
        # Unpack inputs
        y_true, y_pred, input_length, label_length = inputs
        
        # Convert labels to int32 (required by tf.keras.backend.ctc_batch_cost)
        y_true_int = tf.cast(y_true, dtype=tf.int32)

        # Ensure rank 1, then expand to (batch, 1) for CTC
        input_length = tf.cast(tf.expand_dims(input_length, axis=-1), dtype=tf.int32)
        label_length = tf.cast(tf.expand_dims(label_length, axis=-1), dtype=tf.int32)

        # Compute CTC Loss for each sample in the batch
        # tf.keras.backend.ctc_batch_cost returns a (batch, 1) tensor of per-sample losses
        loss = tf.keras.backend.ctc_batch_cost(
            y_true_int,    # true labels (padded)
            y_pred,        # model predictions (softmax probabilities)
            input_length,  # prediction sequence lengths
            label_length,  # label sequence lengths
        )

        # Average over the batch and register as the model's loss
        # add_loss() connects this to model.compile(optimizer=...) automatically
        self.add_loss(tf.reduce_mean(loss))

        # Return y_pred unchanged — this allows us to attach the CTC layer
        # at the end of the model and still get predictions for inference
        return y_pred

    def compute_output_shape(self, input_shape):
        """
        Return the output shape of this layer.
        
        Since we pass through y_pred unchanged (the 2nd input),
        the output shape equals y_pred's shape (input_shape[1]).
        
        input_shape is a list of 4 shapes:
          [y_true_shape, y_pred_shape, input_length_shape, label_length_shape]
        """
        return input_shape[1]


# ── Standalone CTC Decoder (for inference only) ───────────────────
def ctc_decode_greedy(y_pred: tf.Tensor, input_length: tf.Tensor) -> list:
    """
    Greedy CTC decoder — picks the most probable character at each time step.

    WHY: After training, we need to convert the model's raw output
    (probability matrix) into readable text. CTC decoding does this.

    GREEDY vs BEAM SEARCH:
    - Greedy : Fastest, picks top-1 class per time step, then collapses.
               Good enough for most OCR tasks. Used here.
    - Beam   : More accurate, explores multiple paths, much slower.
               Implemented in inference/decoder.py as an option.

    HOW GREEDY DECODING WORKS (Step by Step):
    ------------------------------------------
    1. At each time step, pick the class with highest probability.
       E.g.  time_steps → [c, c, B, a, a, B, t]   (B = blank)

    2. COLLAPSE STEP: Remove consecutive duplicates:
       [c, c, B, a, a, B, t] → [c, B, a, B, t]

    3. REMOVE BLANKS: Delete all blank tokens:
       [c, B, a, B, t] → [c, a, t]  →  "cat"

    Parameters
    ----------
    y_pred       : tf.Tensor  shape (batch, time_steps, num_classes)
    input_length : tf.Tensor  shape (batch,)

    Returns
    -------
    list of lists of int  — decoded integer sequences (one per sample)
    """
    input_length = tf.cast(input_length, dtype=tf.int64)

    # tf.keras.backend.ctc_decode returns a list of decoded sequences
    # (one list element per beam) and log probabilities
    decoded, _ = tf.keras.backend.ctc_decode(
        y_pred,
        input_length=input_length,
        greedy=True,          # Use greedy decoding
        beam_width=1,         # beam_width=1 means greedy
    )

    # decoded[0] has shape (batch, max_decoded_length)
    # Convert to a list of integer lists for easy processing
    results = []
    decoded_dense = decoded[0].numpy()

    for seq in decoded_dense:
        # Filter out -1 padding values added by TF's ctc_decode
        filtered = [int(idx) for idx in seq if idx != -1]
        results.append(filtered)

    return results
