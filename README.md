# IMG_Caption

A small Flask web app that captions an uploaded photo: upload an image, get a
one-sentence description back.

![Upload form](assets/demo_upload_form.png)
![Predicted caption](assets/demo_prediction.png)

> Real output from this model: **"a brown dog is running across the grass"**

## How it works

- **Encoder:** ResNet50 (ImageNet-pretrained, average-pooled), producing a
  2048-d feature vector per image.
- **Decoder:** an LSTM language model, conditioned on the image feature vector
  through a fused image/language branch, trained on the
  [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset
  (~8,000 images, 5 captions each).
- **Decoding:** greedy (argmax), one word at a time, starting from a
  `startofseq` token and stopping at `endofseq` or after 20 words.

```
image ──ResNet50──▶ feature (2048)
                         │
                         ▼
                  Dense + RepeatVector ──┐
                                         ├──▶ concat ──▶ LSTM(128) ──▶ LSTM(512) ──▶ Dense(vocab) ──▶ softmax
caption tokens ──Embedding──▶ LSTM(256) ─┘
```

## Repo structure

```
app.py              # Flask app: model definition, caption decoding, routes
templates/           # index (upload form) / after (result) pages
static/styles.css
vocab.npy            # word<->index mapping the trained model expects
```

`mine_model_weights.h5` (the trained weights, ~19 MB) isn't committed here —
see below.

## Running it

```bash
pip install -r requirements.txt
python app.py                    # http://127.0.0.1:5000
python app.py --port 8000 --debug
```

You'll need `mine_model_weights.h5` in the repo root (trained weights for the
architecture above) and `vocab.npy` (included). Without the weights, `app.py`
will fail at startup — retrain on Flickr8k with this same architecture, or
plug in your own weights file with `--weights path/to/file.h5`.

## Notes on this cleanup

This is a reorganized version of the original 2023 coursework project. Changes
from the original single-script version:

- Wrapped model construction, vocab loading, and Flask routes in a
  `create_app()` factory instead of running at import time.
- Fixed the caption-decoding loop: the original used
  `while tqdm(count < 20):`, which doesn't actually stop the loop the way it
  looks like it should (`tqdm(...)` wraps the boolean into a truthy object,
  so the loop condition never evaluates the count). Replaced with a plain
  bounded loop that breaks cleanly on the `endofseq` token.
- Removed an unused duplicate `import cv2` and an unused `np_utils` import.
- Added a CLI (`--host`, `--port`, `--vocab`, `--weights`, `--debug`) instead
  of hardcoded paths.

The model architecture and training were not changed — this is a
reorganization pass, not a retrain. Your CV lists this project as VGG16-based;
the code that was actually saved uses a ResNet50 encoder — worth a quick check
on which one matches what you actually trained, since the CV wording might be
out of date.

## License

MIT
