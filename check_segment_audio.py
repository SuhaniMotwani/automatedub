from transcriber import transcribe
from synthesizer import synthesize_segments

segments = transcribe('samples/input/QU1Fk-XzT-A.wav')

# Find the two segments you noticed the issue in, by their start times
targets = [s for s in segments if abs(s['start'] - 20.27) < 0.5 or abs(s['start'] - 29.39) < 0.5]

for t in targets:
    print(t)

result = synthesize_segments(targets, 'samples/output/test_run', gender='male')
for r in result:
    print(r)