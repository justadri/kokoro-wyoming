This fork of [@nordwestt](https://github.com/nordwestt)'s [excellent work](https://github.com/nordwestt/kokoro-wyoming) 
brings [Kokoro FastAPI](https://github.com/remsky/Kokoro-FastAPI), a blazing fast API implementation of the original 
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech (TTS) model that can run with GPU acceleration 
to achieve highly realistic speech synthesis, to the [Wyoming Protocol](https://github.com/rhasspy/wyoming) for 
[Home Assistant](https://home-assistant.io).

Kokoro-FastAPI runs in its own container, and kokoro-wyoming runs in another container,  provides the Wyoming Protocol 
wrapper around it. In theory, kokoro-wyoming could be easily extended to work with any TTS engine API.
