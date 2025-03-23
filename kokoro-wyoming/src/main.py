#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from functools import partial

import aiohttp
import numpy as np
import requests
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, TtsProgram, TtsVoice, TtsVoiceSpeaker, Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.server import AsyncServer
from wyoming.tts import Synthesize

_LOGGER = logging.getLogger(__name__)
VERSION = "0.2"


# def split_into_sentences(text: str) -> list[str]:
#     """
#     Split text into sentences using punctuation boundaries.
#
#     Args:
#         text: Input text to split
#
#     Returns:
#         List of sentences
#
#     Example:
#         >>> text = "Hello world! How are you? I'm doing great."
#         >>> split_into_sentences(text)
#         ['Hello world!', 'How are you?', "I'm doing great."]
#     """
#     # First normalize whitespace and clean the text
#     text = ' '.join(text.strip().split())
#
#     # Split on sentence boundaries
#     pattern = r'(?<=[.!?])\s+'
#     sentences = re.split(pattern, text)
#
#     # Filter out empty strings and strip whitespace
#     sentences = [s.strip() for s in sentences if s.strip()]
#
#     return sentences


@dataclass
class KokoroVoice:
    name: str
    language: str
    kokoro_id: str


class KokoroEventHandler(AsyncEventHandler):
    def __init__(self,
                 wyoming_info: Info,
                 kokoro_endpoint,
                 cli_args: argparse.Namespace,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.kokoro_endpoint = kokoro_endpoint
        self.cli_args = cli_args
        self.args = args
        self.wyoming_info_event = wyoming_info.event()
        self.sample_rate = 24000  # Known sample rate for Kokoro
        self.channels = 1
        self.sample_width = 2
        self.chunk_size = 512
        self.speed = 1.0
        # self.normalization_options = args["normalization"]

    async def handle_event(self, event: Event) -> bool:
        """Handle Wyoming protocol events."""
        _LOGGER.debug(f"Handling an Event: {event}")

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if not Synthesize.is_type(event.type):
            _LOGGER.warning("Unexpected event: %s", event)
            return True

        try:
            return await self._handle_synthesize(event)
        except Exception as err:
            await self.write_event(
                Error(text=str(err), code=err.__class__.__name__).event()
            )
            raise err

    async def _handle_synthesize(self, event: Event) -> bool:
        """Handle text to speech synthesis request."""
        synthesize = Synthesize.from_event(event)

        # Get voice settings
        voice_name = "af_heart"  # default voice
        # lang_code = "en"
        if synthesize.voice:
            voice_name = synthesize.voice.name
            # lang_code = synthesize.voice.language

        _LOGGER.info("Starting TTS stream request...")
        start_time = time.time()

        # Initialize variables
        audio_started = False
        chunk_count = 0
        total_bytes = 0
        success = False

        # Make streaming request to API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url=self.kokoro_endpoint,
                        json={
                            "model": "kokoro",
                            "input": synthesize.text,
                            "voice": voice_name,
                            # "lang_code": lang_code,
                            "speed": self.speed,
                            # "normalization_options": self.normalization_options,
                            "response_format": "pcm",
                            "stream": True,
                        },
                        timeout=1800,
                ) as response:

                    response.raise_for_status()
                    _LOGGER.debug(f"Request started successfully after {time.time() - start_time:.2f}s")

                    # Process streaming response with smaller chunks for lower latency
                    async for chunk in response.content.iter_chunked(
                            self.chunk_size):  # 512 bytes = 256 samples at 16-bit
                        if chunk:
                            chunk_count += 1
                            total_bytes += len(chunk)

                            # Handle first chunk
                            if not audio_started:
                                first_chunk_time = time.time()
                                _LOGGER.debug(
                                    f"Received first chunk after {first_chunk_time - start_time:.2f}s"
                                )
                                _LOGGER.debug(f"First chunk size: {len(chunk)} bytes")
                                audio_started = True

                                _LOGGER.debug("Sending AudioStart")
                                await self.write_event(
                                    AudioStart(
                                        rate=self.sample_rate,
                                        width=self.sample_width,
                                        channels=self.channels,
                                    ).event()
                                )

                            # pad last chunk if necessary
                            if len(chunk) < self.chunk_size:
                                padding = self.chunk_size - len(chunk)
                                chunk = chunk + b"\x00" * padding

                            # Convert bytes to numpy array and play
                            # audio_chunk = np.frombuffer(chunk, dtype=np.int16)
                            # audio_bytes = audio_chunk.tobytes()

                            # Convert float32 to int16
                            audio_int16 = np.frombuffer((chunk * 32767), dtype=np.int16)
                            audio_bytes = audio_int16.tobytes()

                            # Send audio chunk
                            await self.write_event(
                                AudioChunk(
                                    audio=audio_bytes,
                                    rate=self.sample_rate,
                                    width=self.sample_width,
                                    channels=self.channels,
                                ).event()
                            )

                            # Log progress every 10 chunks
                            if chunk_count % 100 == 0:
                                elapsed = time.time() - start_time
                                _LOGGER.debug(
                                    f"Progress: {chunk_count} chunks, {total_bytes / 1024:.1f}KB received, {elapsed:.1f}s elapsed"
                                )

            # Final stats
            total_time = time.time() - start_time
            _LOGGER.info(f"Stream complete:")
            _LOGGER.debug(f"Total chunks: {chunk_count}")
            _LOGGER.debug(f"Total data: {total_bytes / 1024:.1f}KB")
            _LOGGER.info(f"Total time: {total_time:.2f}s")
            _LOGGER.info(f"Average speed: {(total_bytes / 1024) / total_time:.1f}KB/s")

            # Clean up
            success = True

        # except requests.exceptions.ConnectionError as e:
        #     _LOGGER.error(f"Connection error - Is the server running? Error: {str(e)}")
        except Exception as e:
            _LOGGER.error(f"Error during streaming: {str(e)}")
        finally:
            # Send audio stop
            _LOGGER.debug("Sending AudioStop")
            await self.write_event(
                AudioStop().event()
            )
            return success


async def main():
    """Main entry point."""
    kokoro_api_host = os.getenv("API_HOST", "http://heracles.dgtlu.net")
    kokoro_api_port = os.getenv("API_PORT", "8880")
    kokoro_endpoint = f"{kokoro_api_host}:{kokoro_api_port}/v1/audio"

    _LOGGER.debug(f"using {kokoro_endpoint} as endpoint")

    listen_host = os.getenv("LISTEN_HOST", "0.0.0.0")
    listen_port = os.getenv("LISTEN_PORT", 10200)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=listen_host,
        help="Host to listen on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=listen_port,
        help="Port to listen on"
    )
    parser.add_argument(
        "--uri",
        default=f"{os.getenv('LISTEN_PROTOCOL', 'tcp')}://{listen_host}:{listen_port}",
        help="unix:// or tcp://"
    )
    parser.add_argument(
        "--speed",
        default=os.getenv("VOICE_SPEED", 1),
        help="Voice speed"
    )
    # parser.add_argument(
    #     "--normalization",
    #     default=os.getenv("VOICE_NORMALIZATION_OPTIONS", ""),
    #     help="Normalization options"
    # )
    parser.add_argument(
        "--debug",
        default=(os.getenv("DEBUG", "false").lower() == "true"),
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Get list of voices from Kokoro endpoint
    response = requests.get(f"{kokoro_endpoint}/voices")
    voice_names = response.json()["voices"]
    voice_names.append("af_heart(2)+af_bella(1)+af_nicole(1)")

    # Define available voices
    voices = [
        TtsVoice(
            name=voice,
            description=f"Kokoro voice {voice}",
            attribution=Attribution(
                name="hexgrad", url="https://github.com/hexgrad/kokoro"
            ),
            installed=True,
            version=None,
            languages=[
                "ja" if voice.startswith("j") else  # japanese
                "zh" if voice.startswith("z") else  # mandarin chinese
                "es" if voice.startswith("e") else  # spanish
                "fr" if voice.startswith("f") else  # french
                "hi" if voice.startswith("h") else  # hindi
                "it" if voice.startswith("i") else  # italian
                "pt" if voice.startswith("p") else  # brazilian portuguese
                "en"  # british and american english
            ],
            speakers=[
                TtsVoiceSpeaker(name=voice.split("_")[1])
            ]
        )
        for voice in voice_names
    ]

    wyoming_info = Info(
        tts=[TtsProgram(
            name="kokoro",
            description="A fast, local, kokoro-based tts engine",
            attribution=Attribution(
                name="Kokoro TTS",
                url="https://huggingface.co/hexgrad/Kokoro-82M",
            ),
            installed=True,
            voices=sorted(voices, key=lambda v: v.name),
            version="1.6.0"
        )]
    )

    server = AsyncServer.from_uri(args.uri)

    # Start server with kokoro instance
    await server.run(partial(KokoroEventHandler, wyoming_info, f"{kokoro_endpoint}/speech", args))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
