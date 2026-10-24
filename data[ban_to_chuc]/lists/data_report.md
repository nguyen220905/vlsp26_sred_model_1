# Data report (train.txt)

- utterances: 58404, speakers: 1900, total hours: 145.9
- sample rates: {16000: 58404}, channels: {1: 58404}, subtypes: {'PCM_16': 58404}
- duration quantiles (s): {0.0: 1.0, 0.01: 1.11, 0.05: 1.57, 0.1: 2.11, 0.25: 3.73, 0.5: 7.36, 0.75: 12.52, 0.9: 18.8, 0.95: 23.61, 0.99: 28.5, 1.0: 29.99}
  - 0-2 s: 8.9%
  - 2-4 s: 18.3%
  - 4-6 s: 14.8%
  - 6-10 s: 21.7%
  - 10-20 s: 27.7%
  - 20-inf s: 8.6%
- utts/speaker quantiles: {0.0: 1.0, 0.1: 4.0, 0.25: 6.0, 0.5: 11.0, 0.75: 41.0, 0.9: 68.0, 0.99: 299.03, 1.0: 577.0}
- exact duplicate files (sha1): 6 files in 3 groups, 2 groups span >1 speaker
- clipping (>0.1% samples at full scale): 5.92% of utts
- crude SNR quantiles (dB): {0.05: 13.9, 0.25: 22.3, 0.5: 29.5, 0.75: 37.2, 0.95: 59.1}
- 99%-energy bandwidth quantiles (Hz): {0.05: 1276.0, 0.25: 2054.0, 0.5: 2770.0, 0.75: 3735.0, 0.95: 5790.0}
- narrow-band (bw < 3.5 kHz): 69.7% of utts, 85.7% of speakers (median)

Duplicate groups spanning several speakers (first 30):
  - id02966/0025.wav, id02665/1690.wav
  - id03180/0132.wav, id02665/0882.wav
