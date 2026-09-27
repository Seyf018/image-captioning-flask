"""IMG_Caption — Flask web app for image captioning.

Encoder: ResNet50 (ImageNet-pretrained, average-pooled features).
Decoder: an LSTM language model conditioned on the image feature vector,
trained on Flickr8k image/caption pairs. Captioning is greedy
(argmax) decoding, one word at a time, until an end-of-sequence token
or a maximum length is reached.

Run with:
    python app.py
then open http://127.0.0.1:5000
"""

import argparse

import cv2
import numpy as np
from flask import Flask, render_template, request
from keras.applications import ResNet50
from keras.layers import LSTM, Activation, Concatenate, Dense, Embedding, RepeatVector, TimeDistributed
from keras.models import Model, Sequential
from keras.utils import pad_sequences

EMBEDDING_SIZE = 128
MAX_CAPTION_LEN = 40  # LSTM input sequence length the model was trained with
MAX_CAPTION_WORDS = 20  # decoding stops after this many words even without an end token
IMAGE_SIZE = 224


def build_model(vocab_size: int) -> Model:
    """Reconstructs the trained captioning model (image encoder projection +
    LSTM language model, fused and decoded jointly)."""
    image_model = Sequential([
        Dense(EMBEDDING_SIZE, input_shape=(2048,), activation="relu"),
        RepeatVector(MAX_CAPTION_LEN),
    ])

    language_model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_SIZE, input_length=MAX_CAPTION_LEN),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(EMBEDDING_SIZE)),
    ])

    merged = Concatenate()([image_model.output, language_model.output])
    x = LSTM(128, return_sequences=True)(merged)
    x = LSTM(512, return_sequences=False)(x)
    x = Dense(vocab_size)(x)
    out = Activation("softmax")(x)

    model = Model(inputs=[image_model.input, language_model.input], outputs=out)
    model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])
    return model


def load_vocab(path: str):
    vocab = np.load(path, allow_pickle=True).item()
    inv_vocab = {index: word for word, index in vocab.items()}
    return vocab, inv_vocab


def generate_caption(model, resnet, vocab, inv_vocab, image_path: str) -> str:
    """Extracts ResNet50 features for `image_path` and greedily decodes a
    caption word by word until 'endofseq' or MAX_CAPTION_LEN is reached."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

    features = resnet.predict(image).reshape(1, 2048)

    text_in = ["startofseq"]
    caption_words = []

    for _ in range(MAX_CAPTION_WORDS):
        encoded = [vocab[word] for word in text_in]
        padded = pad_sequences([encoded], maxlen=MAX_CAPTION_LEN, padding="post", truncating="post")
        padded = padded.reshape(1, MAX_CAPTION_LEN)

        next_index = np.argmax(model.predict([features, padded], verbose=0))
        next_word = inv_vocab[next_index]

        if next_word == "endofseq":
            break

        caption_words.append(next_word)
        text_in.append(next_word)

    return " ".join(caption_words)


def create_app(vocab_path="vocab.npy", weights_path="mine_model_weights.h5") -> Flask:
    vocab, inv_vocab = load_vocab(vocab_path)

    model = build_model(vocab_size=len(vocab))
    model.load_weights(weights_path)

    resnet = ResNet50(include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), pooling="avg")

    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/after", methods=["POST"])
    def after():
        uploaded = request.files["file1"]
        image_path = "static/file.jpg"
        uploaded.save(image_path)

        caption = generate_caption(model, resnet, vocab, inv_vocab, image_path)
        return render_template("after.html", data=caption)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMG_Caption Flask app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--vocab", default="vocab.npy")
    parser.add_argument("--weights", default="mine_model_weights.h5")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    flask_app = create_app(args.vocab, args.weights)
    flask_app.run(host=args.host, port=args.port, debug=args.debug)
