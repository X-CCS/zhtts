import os
import numpy as np
from pathlib import Path
import tensorflow as tf 
#import tflite_runtime.interpreter as tflite
from scipy.io import wavfile
import re
import sys
sys.path.append("/home/project/zhtts/")
from tensorflow_tts.processor import BakerProcessor # baker.py

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ASSET_DIR = Path(__file__).parent / "asset"

def split_sens(text):
    """ split sentence and keep sperator to the left 

    Args:
        text (str): 
        
    Returns:
        list[str]: splited sentence
        
    Examples:
        >>> split_sens("中文：语音，合成！系统\n")
        ['中文：', '语音，', '合成！', '系统']
    """
    texts = re.split(r";", re.sub(r"([、，。！？])", r"\1;", text.strip()))
    return [x for x in texts if x]

class TTS():
    def __init__(self, text2mel_name="FASTSPEECH2"):
        """text2mel_name: ["FASTSPEECH2", "TACOTRON"] """
        self.sample_rate = 24000
        self.processor = BakerProcessor(
            data_dir=None, loaded_mapper_path=ASSET_DIR / "baker_mapper.json")
        self.text2mel_name = text2mel_name
        if text2mel_name == "FASTSPEECH2":
            # fastspeech2_quan.tflite
            self.acoustic = tf.lite.Interpreter(model_path=str(ASSET_DIR / 'fastspeech2_test_quan.tflite'))
        elif text2mel_name == "TACOTRON":
            # tacotron2_quan.tflite
            self.acoustic = tf.lite.Interpreter(model_path=str(ASSET_DIR / 'tacotron2_quan.tflite'))
        else:
            raise ValueError(f"unsported text2mel_name: {text2mel_name}")
        # mb_melgan.tflite
        self.vocoder = tf.lite.Interpreter(model_path=str(ASSET_DIR / 'mb_melgan_test.tflite'))

    def prepare_input(self, input_ids):
        input_ids = np.expand_dims(np.array(input_ids, np.int32), 0)
        if self.text2mel_name == "TACOTRON":
            return (input_ids,
                    np.array([input_ids.shape[1]], np.int32),
                    np.array([0], np.int32),)
        elif self.text2mel_name == "FASTSPEECH2":
            return (input_ids,
                    np.array([0], np.int32),
                    np.array([1.0], np.float32),
                    np.array([1.0], np.float32),
                    np.array([1.0], np.float32),)

    def text2mel(self, input_text):
        input_details = self.acoustic.get_input_details() 
        output_details = self.acoustic.get_output_details()
        input_ids = self.processor.text_to_sequence(input_text, inference=True)
        
        self.acoustic.resize_tensor_input(
            input_details[0]['index'], [1, len(input_ids)])
        self.acoustic.allocate_tensors()

        input_data = self.prepare_input(input_ids)
        for i, detail in enumerate(input_details):
            self.acoustic.set_tensor(detail['index'], input_data[i])
        self.acoustic.invoke()

        return self.acoustic.get_tensor(output_details[1]['index'])

    def mel2audio(self, mel):
        input_details = self.vocoder.get_input_details()
        output_details = self.vocoder.get_output_details()
        self.vocoder.resize_tensor_input(input_details[0]['index'], mel.shape)
        self.vocoder.allocate_tensors()
        self.vocoder.set_tensor(input_details[0]['index'], mel)
        self.vocoder.invoke()

        return self.vocoder.get_tensor(output_details[0]['index'])[0, :, 0]

    def synthesis(self, text, sil_time=0.2):
        """ synthesis text to audio

        Args:
            text (str)
            sil_time (float): silence duration between two wav 两个wav之间的沉默时间
        Returns:
            ndarray: audio
        """
        audios = []
        texts = split_sens(text)
        silence = np.zeros(int(sil_time * self.sample_rate), dtype=np.float32) # 添加静音
        for i, text in enumerate(texts):
            print(f"index: {i}, text: {text}")
            print(f"frontend info: {self.frontend(text)}")
            # print(self.processor.text_to_sequence(text, inference=True))
            mel = self.text2mel(text)
            # np.save("mel_tacotron.npy",mel)
            # np.save("mel_fastseppch2.npy",mel)
            # print("mel内容:",mel)
            # print("mel类型:",type(mel))
            # print("mel维度:",mel.shape) # fastspeech2(1, 259, 80) tacotorn2(1, 290, 80)

            audio = self.mel2audio(mel)
            np.save("audio_tacotron.npy",audio)
            # np.save("audio_fastseppch2.npy",audio)
            print("audio内容:",audio)
            print("audio类型:",type(audio))
            print("audio维度:",audio.shape) # fastspeech2(77700,) tacotorn2(85800,)
            if self.text2mel_name == "TACOTRON":
                audio = audio[:-2048]  # tacotron will generate noise at the end
            audios.append(audio)
            if i < len(texts)-1:
                audios.append(silence)
        return np.concatenate(audios) # 数组拼接函数

    def frontend(self, text):
        """ return normalize_text, phoneme_seq for debug

        Args:
            text (str)
        Returns:
            (tuple): tuple containing:

                normalize_text (str):  text after text_normalize
                phoneme (str):  " ".join(phones)
        """
        return self.processor.text_to_phone(text)

    def text2wav(self, text, wavpath):
        """synthesis text and save to wavfile"""
        audio = self.synthesis(text)
        
        wavfile.write(wavpath, self.sample_rate, audio)
        print(f"Save wav to {wavpath}")

if __name__ == "__main__": 
    tts = TTS()
    # tts = TTS(text2mel_name="TACOTRON") # tacotron2
    tts = TTS(text2mel_name="FASTSPEECH2") # FASTSPEECH2
    # text = "2020年，这是一个开源的端到端中文语音合成系统"
    # res = tts.frontend(text)
    # tts.synthesis(text)
    tts.text2wav(text,"./test.wav")
    # print(res)
    print("done")