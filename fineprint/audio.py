from pydub import AudioSegment
import numpy as np

AUDIO_SUPPORTED = ('.mp3', '.wav', )

CHANNEL_DICT = {
  'mono': 1,
  'stereo': 2
}

class FpAudio:
  def __init__(self, audio_file: str, channel: str = 'mono', freq: int = 5512, limit: int = None):
    """
    Audio file to consider for fingerprinting.
    :param audio_file: path to the audio file
    :param channel: mono/stereo
    :param limit: audio file to consider from 0 to `limit` seconds.
    """
    assert audio_file.lower().endswith(AUDIO_SUPPORTED), "Currently only supports MP3 & WAV files."
    assert (channel.lower() == 'mono') or (channel.lower() == 'stereo'), "Channel can only be mono or stereo."
    self.audio = AudioSegment.from_file(audio_file).set_channels(CHANNEL_DICT[channel])
    
    if limit:
      self.audio = self.audio[:1000 * limit]
    if freq:
      self.audio = self.audio.set_frame_rate(freq)
    
    self.data = np.frombuffer(self.audio.raw_data, dtype=np.int16)  # PCM
    
    assert self.data.shape[0] == self.audio.channels * self.audio.frame_rate * self.audio.duration_seconds,\
    "Something went wrong. Converted PCM data isn't reflecting your audio."


if __name__ == '__main__':
  from matplotlib import pyplot as plt
  fpaudio = FpAudio("../song/kasoor.mp3")
  plt.plot(fpaudio.data)
  plt.show()
