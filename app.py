import os
import time
import tempfile
import logging
import re
from datetime import timedelta
from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment
import concurrent.futures
from openai import OpenAI
import srt
from aeneas.task import Task
from aeneas.executetask import ExecuteTask

# ----- Configuration -----
CHUNK_MS = 4 * 60 * 1000  # 4 minutes in milliseconds
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB OpenAI limit

# ----- Logging Setup -----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----- OpenAI Client -----
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger.info("OpenAI client initialized")

# ----- Flask App -----
app = Flask(__name__)


def chunk_audio(path: str) -> list[tuple[str, float]]:
    """Split audio file into chunks that stay below API size limits."""
    logger.info(f"Chunking audio: {path}")
    audio = AudioSegment.from_file(path)
    length = len(audio)
    chunks = []
    for i, start in enumerate(range(0, length, CHUNK_MS)):
        end = min(start + CHUNK_MS, length)
        segment = audio[start:end]
        fn = os.path.join(tempfile.gettempdir(), f"chunk_{i}.wav")
        segment.export(fn, format="wav")
        if os.path.getsize(fn) > MAX_FILE_SIZE_BYTES:
            os.unlink(fn)
            subchunk_ms = (end - start) // 2
            for j, substart in enumerate(range(start, end, subchunk_ms)):
                subend = min(substart + subchunk_ms, end)
                subsegment = audio[substart:subend]
                subfn = os.path.join(tempfile.gettempdir(), f"chunk_{i}_{j}.wav")
                subsegment.export(subfn, format="wav")
                chunks.append((subfn, substart / 1000.0))
        else:
            chunks.append((fn, start / 1000.0))
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks


def run_transcription(transcribe_fn, chunks, language: str):
    """
    Run a transcription function over all chunks in parallel.
    Returns list of {output, offset} and max runtime
    """
    results = []
    max_runtime = 0.0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_map = {executor.submit(transcribe_fn, path, offset, language): (path, offset)
                      for path, offset in chunks}
        for future in concurrent.futures.as_completed(future_map):
            output, rt = future.result()
            results.append({"output": output, "offset": future_map[future][1]})
            max_runtime = max(max_runtime, rt)
    return results, max_runtime


def _transcribe_4o_chunk(path: str, offset: float, language: str):
    """Transcribe a chunk using gpt-4o-mini-transcribe."""
    t0 = time.time()
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            language=language
        )
    return resp.text.strip(), time.time() - t0


def process_4o_stream(chunks, audio_path: str, language: str):
    """
    Process the 4o-mini stream: transcription parallel + Aeneas alignment
    Returns {srt_path, runtime}
    """
    logger.info(f"Processing 4o-mini stream with {len(chunks)} chunks")
    transcripts, runtime = run_transcription(_transcribe_4o_chunk, chunks, language)
    transcripts.sort(key=lambda x: x['offset'])

    # Write raw transcript
    transcript_txt = os.path.join(tempfile.gettempdir(), "4o_full.txt")
    with open(transcript_txt, "w", encoding="utf-8") as fw:
        for item in transcripts:
            fw.write(item['output'] + ' ')

    # Sentence-split transcript
    sentences_txt = os.path.join(tempfile.gettempdir(), "4o_sentences.txt")
    with open(transcript_txt, "r", encoding="utf-8") as fr:
        text = fr.read()
    split_text = re.sub(r'([\.\?!])\s+', r"\1\n", text)
    with open(sentences_txt, "w", encoding="utf-8") as fw:
        fw.write(split_text)
    logger.info(f"Sentence-split transcript: {sentences_txt}")

    # Aeneas alignment
    srt_path = os.path.join(tempfile.gettempdir(), "4o_alignment.srt")
    config = (
        f"task_language={language}|"
        "is_text_type=plain|"
        "os_task_file_format=srt|"
        "task_adjust_boundary_rate_value=10.0|"
        "task_adjust_boundary_nonspeech_min=0.2"
    )
    task = Task(config_string=config)
    task.audio_file_path_absolute    = audio_path
    task.text_file_path_absolute     = sentences_txt
    task.sync_map_file_path_absolute = srt_path

    ExecuteTask(task).execute()
    task.output_sync_map_file()
    logger.info(f"Aeneas SRT written to {srt_path}")

    return {"srt_path": srt_path, "runtime": runtime}


def _transcribe_whisper_chunk(path: str, offset: float, language: str):
    """Transcribe chunk using Whisper with verbose JSON."""
    t0 = time.time()
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            language=language
        )
    subs = [srt.Subtitle(index=seg.id,
                         start=timedelta(seconds=seg.start + offset),
                         end=timedelta(seconds=seg.end + offset),
                         content=seg.text.strip())
            for seg in resp.segments]
    return subs, time.time() - t0


def process_whisper_stream(chunks, audio_path: str, language: str):
    """Parallel Whisper transcription + shift timestamps."""
    results, runtime = run_transcription(_transcribe_whisper_chunk, chunks, language)
    all_subs = []
    for r in results:
        all_subs.extend(r['output'])
    merged = srt.compose(sorted(all_subs, key=lambda x: x.start))
    srt_path = os.path.join(tempfile.gettempdir(), "whisper_full.srt")
    with open(srt_path, "w", encoding="utf-8") as fw:
        fw.write(merged)
    logger.info(f"Whisper SRT written to {srt_path}")
    return {"srt_path": srt_path, "runtime": runtime}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/compare", methods=["POST"])
def compare():
    audio_file = request.files['audio']
    language   = request.form['language']
    tmp = tempfile.NamedTemporaryFile(delete=False,
                                     suffix=os.path.splitext(audio_file.filename)[1])
    audio_file.save(tmp.name)
    chunks = chunk_audio(tmp.name)

    # 4o-mini
    try:
        res4 = process_4o_stream(chunks, tmp.name, language)
        srt4 = open(res4['srt_path'], 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"4o-mini failed: {e}")
        srt4, res4 = f"Error: {e}", {'runtime': 0}

    # Whisper
    try:
        resw = process_whisper_stream(chunks, tmp.name, language)
        srtw = open(resw['srt_path'], 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f"Whisper failed: {e}")
        srtw, resw = f"Error: {e}", {'runtime': 0}

    # Cleanup
    os.unlink(tmp.name)
    for fn, _ in chunks:
        os.unlink(fn)

    return jsonify({
        'four_o_srt':      srt4,
        'four_o_runtime':  res4['runtime'],
        'whisper_srt':     srtw,
        'whisper_runtime': resw['runtime'],
    })


if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(debug=True)
