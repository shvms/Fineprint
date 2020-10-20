import os
from scipy.io import wavfile
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List


class DataGenerator:
  """
  Takes in a list of audio files and generates a list of audio files
  in segments for training.
  """
  
  def __init__(self, files: List[str]):
    self.files = files
  
  def generate(self, channel: str, segment_size: int = 8, output_dir: str = 'output'):
    assert channel in ['left', 'right', 'mono'], "channel should either 'left', 'right' or mono."
    
    for file in self.files:
      print(f"For {file}...")
      fs, data = wavfile.read(file)
      
      if channel == 'left':
        data = data[:, 0]
      elif channel == 'right':
        data = data[:, 1]
      else:
        data = np.mean(data, axis=1)
      
      receptive_fields = DataGenerator.__gen_receptive_fields(data, fs, segment_size)
      for i in tqdm(range(len(receptive_fields))):
        DataGenerator.__gen_spectrogram(receptive_fields[i], fs,
                                        os.path.join(output_dir, os.path.basename(file),
                                                     f"{str(i+1)}.png"))
  
  @staticmethod
  def __gen_receptive_fields(data: np.ndarray, fs: int, segment_size: int) -> List[np.ndarray]:
    """
    Generate audio receptive fields, i.e., slides a window of segment_size over
    the audio file to generate `segment_size` audio files.
    :param data: Data read from WAV file
    :param fs: Sampling rate of audio (in Hz)
    :param segment_size: Segment size (in seconds)
    :return: Returns an list of receptive fields inferred from the given audio
    """
    window = segment_size * fs
    receptive_fields = []
    
    for i in range(0, data.shape[0] - window, fs):
      receptive_fields.append(data[i:i+window])
    
    return receptive_fields
  
  @staticmethod
  def __gen_spectrogram(data: np.ndarray, fs: int, output_path: str):
    plt.specgram(data, Fs=fs)
    plt.savefig(output_path, bbox_inches='tight')


if __name__ == '__main__':
  files = ["../song/bella_ciao_1.wav"]
  generator = DataGenerator(files)
  generator.generate(channel='mono', output_dir="../output")
