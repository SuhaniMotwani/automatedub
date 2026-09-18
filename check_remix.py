from transcriber import transcribe
from synthesizer import synthesize_segments
from remixer import remix

segments = transcribe('samples/input/QU1Fk-XzT-A.wav')[:5]
segments = synthesize_segments(segments, 'samples/output/test_run', gender='male')
output = remix('samples/input/QU1Fk-XzT-A.mp4', segments, 'samples/output/test_run/remixed_test3.mp4')
print('Output:', output)