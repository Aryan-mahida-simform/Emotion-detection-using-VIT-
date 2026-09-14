# Emotion-detection-using-VIT-

Emotion detection using a Vision Transformer (ViT). The model classifies a face image into one of
seven emotions: angry, disgust, fear, happy, neutral, sad, surprise.

## What is in this repository

This branch carries the browser frontend that shows the model's output. It lives in `frontend/` and
is plain HTML, CSS, and ES modules with no runtime dependencies.

The model service itself is a separate workstream. The frontend reads two response shapes, both of
which it handles today, and `frontend/README.md` documents each one with a full example payload.

## Run the frontend

```bash
cd frontend
npm start
```

That serves the page on <http://localhost:8080> alongside a mock model service, so the whole flow
works without a trained model. See `frontend/README.md` for pointing it at a real service, the
request and response contract, and the test suite.

## Support

Current Chromium, Firefox, and Safari. Webcam capture needs a secure context, so it works on
`localhost` and over HTTPS.
