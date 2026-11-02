# Lamplight

A read-aloud PDF book reader that runs on your own PC.

## What it is

- Shows the real PDF pages, highlights the sentence being read, and follows along.
- Book-page mode with page turns, or continuous scroll.
- Works on a phone, at home or away over Tailscale.
- Remembers your place across devices.

## Requirements

- Windows
- An NVIDIA GPU is recommended. Tested on an RTX 4070 Ti Super. It works on CPU but is slow.
- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Getting started

1. Double-click `Start Book Reader.bat`. The first start downloads the voice models and can take several minutes.
2. Open http://localhost:8765 in a browser.
3. Add books by putting PDFs in the `books/` folder, or with "Add a PDF" in the app.

## Voices

- **Natural**: Chatterbox Turbo by Resemble AI.
- **Fast**: Kokoro-82M.
- To add your own natural voice, put a clean, roughly 10 second `.wav` recording of the voice in `voices/` and restart the app. Only use voices you have the right to use.

## Phone

Open `http://<computer's IP>:8765` in your phone's browser. Use the computer's Wi-Fi IP at home, or its Tailscale IP from anywhere. Windows Firewall may ask whether to allow Python on private networks. Allow it.

## How it works

PyMuPDF extracts each word with its position on the page, then splits the text into sentences. It skips running headers and footers, footnotes, code listings and reference lists. The server synthesizes each sentence when it is needed, and the page prefetches the next few so playback continues without gaps.

## Notes

- The Python environment is created in `%USERPROFILE%\.book-reader-env`, not in the project folder, because long Windows paths break some packages.
- Keep your PDFs in `books/` local. That folder is git-ignored so copyrighted books are never committed.
