import sys
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFAutoModelForMaskedLM
import tensorflow as tf
import numpy as np

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")
    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")  # Change to TensorFlow tensor
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFAutoModelForMaskedLM.from_pretrained(
        MODEL,
        from_pt=True,
        use_safetensors=False,
    )
    result = model(**inputs, output_attentions=True, training=False)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].numpy().tolist(), skip_special_tokens=False
    )
    visualize_attentions(tokens, result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    input_ids = (
        inputs["input_ids"][0].numpy().tolist()
    )  # Get numpy array from TensorFlow tensor
    for i, tid in enumerate(input_ids):
        if int(tid) == int(mask_token_id):
            return i
    return None


def get_color_for_attention_score(attention_score: float):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    score = max(0.0, min(1.0, attention_score))
    gray = round(score * 255)
    return (gray, gray, gray)


def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    # TODO: Update this function to produce diagrams for all layers and heads.
    # attentions is a tuple length 12
    print(f"Tokens: {tokens}")
    print(f"Attentions: {attentions}")
    for layer_index, layer_attention in enumerate(attentions, start=1):
        num_heads = layer_attention.shape[1]

        for head_idx in range(num_heads):
            att = layer_attention[0, head_idx].numpy()  # Convert to NumPy array
            generate_diagram(layer_index, head_idx + 1, tokens, att)


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT,
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT,
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"S2_Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()
