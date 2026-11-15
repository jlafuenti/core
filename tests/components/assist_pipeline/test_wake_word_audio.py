"""Test wake word audio from satellites in Assist pipelines."""

import array
import asyncio
from collections.abc import AsyncGenerator, AsyncIterable
from datetime import timedelta
from unittest.mock import patch

import pytest
from pytest_homeassistant.common import async_fire_time_changed

from homeassistant.components import assist_pipeline, stt, tts
from homeassistant.components.assist_pipeline import PipelineEvent, PipelineEventType
from homeassistant.components.assist_pipeline.arbitration import (
    DATA_ARBITER,
    WakeWordArbitrationMode,
)
from homeassistant.components.assist_pipeline.const import MS_PER_CHUNK
from homeassistant.core import Context, HomeAssistant
from homeassistant.util import dt as dt_util

from .conftest import MockSTTProviderEntity, make_10ms_chunk

WAKE = make_10ms_chunk(b"wake")
WORD = make_10ms_chunk(b"word")
COMMAND = make_10ms_chunk(b"command")


async def _audio_stream(*chunks: bytes) -> AsyncGenerator[bytes]:
    for chunk in chunks:
        yield chunk


async def _async_run_pipeline(
    hass: HomeAssistant,
    audio_stream: AsyncIterable[bytes],
    wake_word_audio_ms: int,
    end_stage: assist_pipeline.PipelineStage = assist_pipeline.PipelineStage.STT,
    start_stage: assist_pipeline.PipelineStage = assist_pipeline.PipelineStage.STT,
    wake_word_phrase: str | None = "okay nabu",
    is_vad_enabled: bool = False,
    satellite_id: str | None = None,
) -> list[PipelineEvent]:
    events: list[PipelineEvent] = []
    await assist_pipeline.async_pipeline_from_audio_stream(
        hass,
        context=Context(),
        event_callback=events.append,
        satellite_id=satellite_id,
        stt_metadata=stt.SpeechMetadata(
            language="",
            format=stt.AudioFormats.WAV,
            codec=stt.AudioCodecs.PCM,
            bit_rate=stt.AudioBitRates.BITRATE_16,
            sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
            channel=stt.AudioChannels.CHANNEL_MONO,
        ),
        stt_stream=audio_stream,
        wake_word_phrase=wake_word_phrase,
        wake_word_audio_ms=wake_word_audio_ms,
        audio_settings=assist_pipeline.AudioSettings(is_vad_enabled=is_vad_enabled),
        start_stage=start_stage,
        end_stage=end_stage,
    )
    return events


@pytest.mark.parametrize(
    ("wake_word_audio_ms", "expected_stt_audio"),
    [
        pytest.param(2 * MS_PER_CHUNK, COMMAND, id="whole_chunks"),
        # Rounded down to whole chunks, so the stream's last chunk isn't cut short
        pytest.param(
            MS_PER_CHUNK + (MS_PER_CHUNK // 2), WORD + COMMAND, id="within_chunk"
        ),
    ],
)
@pytest.mark.usefixtures("init_components")
async def test_wake_word_audio(
    hass: HomeAssistant,
    mock_stt_provider_entity: MockSTTProviderEntity,
    wake_word_audio_ms: int,
    expected_stt_audio: bytes,
) -> None:
    """Test the wake word is accepted and its audio removed for speech-to-text."""
    events = await _async_run_pipeline(
        hass, _audio_stream(WAKE, WORD, COMMAND), wake_word_audio_ms
    )

    assert [event.type for event in events] == [
        PipelineEventType.RUN_START,
        PipelineEventType.WAKE_WORD_END,
        PipelineEventType.STT_START,
        PipelineEventType.STT_END,
        PipelineEventType.RUN_END,
    ]
    assert events[1].data == {
        "wake_word_output": {
            "wake_word_phrase": "okay nabu",
            "wake_word_audio_ms": wake_word_audio_ms,
        }
    }
    assert b"".join(mock_stt_provider_entity.received) == expected_stt_audio


@pytest.mark.parametrize(
    "end_stage",
    [
        pytest.param(assist_pipeline.PipelineStage.STT, id="end_at_stt"),
        pytest.param(assist_pipeline.PipelineStage.TTS, id="end_at_tts"),
    ],
)
@pytest.mark.usefixtures("init_components", "mock_stt_provider_entity")
async def test_wake_word_audio_duplicate(
    hass: HomeAssistant, end_stage: assist_pipeline.PipelineStage
) -> None:
    """Test a wake word another satellite just woke up with ends the run."""
    first_events = await _async_run_pipeline(
        hass, _audio_stream(WAKE, COMMAND), MS_PER_CHUNK
    )
    second_events = await _async_run_pipeline(
        hass, _audio_stream(WAKE, COMMAND), MS_PER_CHUNK, end_stage
    )

    assert first_events[1].type is PipelineEventType.WAKE_WORD_END
    # The same error as for a duplicate without wake word audio
    assert [event.type for event in second_events] == [
        PipelineEventType.RUN_START,
        PipelineEventType.ERROR,
        PipelineEventType.RUN_END,
    ]
    assert second_events[1].data["code"] == "duplicate_wake_up_detected"
    # The text-to-speech stream prepared for the run is removed
    assert not hass.data[tts.DATA_TTS_MANAGER].token_to_stream


@pytest.mark.usefixtures(
    "init_components", "mock_stt_provider_entity", "mock_wake_word_provider_entity"
)
async def test_wake_word_audio_detected_by_pipeline(hass: HomeAssistant) -> None:
    """Test a run starting at wake word detection accepts the wake word once."""
    events = await _async_run_pipeline(
        hass,
        _audio_stream(make_10ms_chunk(b"wake word"), COMMAND),
        MS_PER_CHUNK,
        start_stage=assist_pipeline.PipelineStage.WAKE_WORD,
        # The phrase the mock wake word entity detects
        wake_word_phrase="Test Wake Word",
    )

    assert [event.type for event in events] == [
        PipelineEventType.RUN_START,
        PipelineEventType.WAKE_WORD_START,
        PipelineEventType.WAKE_WORD_END,
        PipelineEventType.STT_START,
        PipelineEventType.STT_END,
        PipelineEventType.RUN_END,
    ]


@pytest.mark.usefixtures("init_components")
async def test_wake_word_audio_requires_phrase(
    hass: HomeAssistant,
    mock_stt_provider_entity: MockSTTProviderEntity,
) -> None:
    """Test wake word audio is rejected without the phrase it belongs to."""
    events = await _async_run_pipeline(
        hass, _audio_stream(WAKE, COMMAND), MS_PER_CHUNK, wake_word_phrase=None
    )

    assert [event.type for event in events] == [
        PipelineEventType.RUN_START,
        PipelineEventType.ERROR,
        PipelineEventType.RUN_END,
    ]
    assert events[1].data == {
        "code": "validation-error",
        "message": "wake_word_phrase is required with wake_word_audio_ms",
    }
    assert not mock_stt_provider_entity.received


@pytest.mark.usefixtures("init_components")
async def test_wake_word_audio_timestamps(
    hass: HomeAssistant,
    mock_stt_provider_entity: MockSTTProviderEntity,
) -> None:
    """Test audio timestamps count from the start of the stream."""
    with patch.object(
        mock_stt_provider_entity,
        "async_get_audio_processing",
        return_value=stt.SpeechAudioProcessing(
            requires_external_vad=False,
            prefers_auto_gain_enabled=True,
            prefers_noise_reduction_enabled=True,
        ),
    ):
        events = await _async_run_pipeline(
            hass,
            _audio_stream(WAKE, WORD, COMMAND),
            2 * MS_PER_CHUNK,
            is_vad_enabled=True,
        )

    # The provider ends the command after the last chunk, which starts 20 ms
    # into the stream, as when the pipeline detects the wake word itself
    assert [
        event.data for event in events if event.type is PipelineEventType.STT_VAD_END
    ] == [{"timestamp": 2 * MS_PER_CHUNK}]


def _level_chunk(amplitude: int) -> bytes:
    """Return 10 ms of audio at a constant level."""
    return array.array("h", [amplitude, -amplitude] * 80).tobytes()


@pytest.mark.usefixtures("init_components")
async def test_wake_word_audio_loudest_satellite(
    hass: HomeAssistant, mock_stt_provider_entity: MockSTTProviderEntity
) -> None:
    """Test the satellite that heard the wake word loudest gets the command."""
    await hass.data[DATA_ARBITER].async_update_settings(
        mode=WakeWordArbitrationMode.LOUDEST, window_ms=400
    )
    quiet_wake = _level_chunk(100)
    loud_wake = _level_chunk(1000)

    # The quiet satellite reports first
    quiet = hass.async_create_task(
        _async_run_pipeline(
            hass,
            _audio_stream(quiet_wake, make_10ms_chunk(b"far")),
            MS_PER_CHUNK,
            satellite_id="assist_satellite.quiet",
        )
    )
    await asyncio.sleep(0)
    loud = hass.async_create_task(
        _async_run_pipeline(
            hass,
            # A chunk holding the end of the wake word and the start of the command
            _audio_stream(loud_wake + COMMAND),
            MS_PER_CHUNK,
            satellite_id="assist_satellite.loud",
        )
    )
    await asyncio.sleep(0)
    assert not quiet.done()
    assert not loud.done()

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(milliseconds=401))
    quiet_events = await quiet
    loud_events = await loud

    assert [event.type for event in quiet_events] == [
        PipelineEventType.RUN_START,
        PipelineEventType.ERROR,
        PipelineEventType.RUN_END,
    ]
    assert quiet_events[1].data["code"] == "duplicate_wake_up_detected"

    assert [event.type for event in loud_events] == [
        PipelineEventType.RUN_START,
        PipelineEventType.WAKE_WORD_END,
        PipelineEventType.STT_START,
        PipelineEventType.STT_END,
        PipelineEventType.RUN_END,
    ]
    arbitration = loud_events[1].data["wake_word_output"]["arbitration"]
    assert arbitration["winner"] == "assist_satellite.loud"
    assert [
        (candidate["satellite_id"], round(candidate["level_db"]))
        for candidate in arbitration["candidates"]
    ] == [("assist_satellite.quiet", -50), ("assist_satellite.loud", -30)]
    # Only the winner's command reaches speech-to-text, without its wake word
    assert b"".join(mock_stt_provider_entity.received) == COMMAND


@pytest.mark.usefixtures("init_components")
async def test_wake_word_audio_loudest_without_audio(
    hass: HomeAssistant, mock_stt_provider_entity: MockSTTProviderEntity
) -> None:
    """Test a satellite that sends no wake word audio waits for the others."""
    await hass.data[DATA_ARBITER].async_update_settings(
        mode=WakeWordArbitrationMode.LOUDEST, window_ms=400
    )
    events: list[PipelineEvent] = []

    async def run_without_wake_word_audio() -> None:
        await assist_pipeline.async_pipeline_from_audio_stream(
            hass,
            context=Context(),
            event_callback=events.append,
            stt_metadata=stt.SpeechMetadata(
                language="",
                format=stt.AudioFormats.WAV,
                codec=stt.AudioCodecs.PCM,
                bit_rate=stt.AudioBitRates.BITRATE_16,
                sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
                channel=stt.AudioChannels.CHANNEL_MONO,
            ),
            stt_stream=_audio_stream(COMMAND),
            wake_word_phrase="okay nabu",
            audio_settings=assist_pipeline.AudioSettings(is_vad_enabled=False),
            end_stage=assist_pipeline.PipelineStage.STT,
        )

    first = hass.async_create_task(run_without_wake_word_audio())
    await asyncio.sleep(0)
    assert not first.done()
    second_events_task = hass.async_create_task(
        _async_run_pipeline(
            hass, _audio_stream(_level_chunk(100), COMMAND), MS_PER_CHUNK
        )
    )
    await asyncio.sleep(0)

    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(milliseconds=401))
    await first
    second_events = await second_events_task

    # The satellite whose wake word could be measured wins
    assert events[1].type is PipelineEventType.ERROR
    assert events[1].data["code"] == "duplicate_wake_up_detected"
    assert second_events[1].type is PipelineEventType.WAKE_WORD_END
    assert b"".join(mock_stt_provider_entity.received) == COMMAND
