# VishGuard Demo

A Gradio web app that detects AI-generated voices in audio recordings using the
trained Mel-CNN ensemble from this project.

## How to run

From the repo root:

```bash
pip install gradio
python3 demo/app.py
```

Then open http://localhost:7860 in your browser.

## What it does

1. Upload a WAV or FLAC audio file
2. Choose the channel condition (clean / telephone / telephone + codec)
3. Choose the training regime and alarm rule
4. Click **Analyse**

The system slides a 1-second window (0.5 s hop) over the recording and scores
each window with the 5-seed Mel-CNN ensemble. It plots the margin above the
frozen detection threshold for every window and raises an alarm if the chosen
rule fires.

## Options

| Option | Values | Description |
|--------|--------|-------------|
| Training regime | `aug` (default), `clean` | `aug` = trained on clean + telephone + codec; more robust |
| Channel condition | `clean`, `telephone`, `telephone_codec` | Simulates the channel applied before scoring |
| Alarm rule | `consecutive` (default), `mean`, `vote` | How windows are aggregated into a final decision |
| k | 1–6 | Number of windows required for the alarm to fire |

## Notes

- Raw audio and processed feature caches are **not** required to run the demo;
  only the saved model files in `outputs/seeds5/models/` are needed.
- Research demonstration only — not intended for production use.
