import pyaudio as po
import soundfile as sf
import wave
import numpy as np

FORMAT = po.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "file.wav"
THRESHOLD = 5
VOLUME = 0.5
SAMPLE_FORMAT = po.paInt16

def record_something(seconds: int) -> None:
    p = po.PyAudio()
    record = p.get_host_api_info_by_index(0)
    num = record.get("deviceCount")
    for i in range(num):
        channel = p.get_device_info_by_host_api_device_index(0,i).get("maxInputChannels")
        if channel == 2:
            index = i
    stream = p.open(format=FORMAT, channels=CHANNELS, input_device_index=index, 
                    rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("..Recording")
    recordFrames = []
    for i in range(int(RATE/CHUNK*seconds)):
        data = stream.read(CHUNK)
        recordFrames.append(data)
    print("Recording stopped") 
    stream.stop_stream()
    stream.close()
    p.terminate()
    byte_array = b''.join(recordFrames)
    int16_array = np.frombuffer(byte_array, dtype=np.int16)
    filename = "./audio.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(recordFrames))
    wf.close()
    return int16_array