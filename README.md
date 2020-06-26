# Fineprint
An audio fingerprinting system based on "*Content fingerprinting using wavelets*" paper. **Under development**

#### Steps involved:
- [x] Preprocessing audio from its standard settings (assumed to be 16-bit stereo) to PCM based mono audio on ~5000Hz.
- [ ] Using FFT to draw spectrogram. Required for band filtering. Helps in reducing noise.
- [ ] Minhashing for fingerprinting.
- [ ] Group LSH labels
- [ ] Database storage
