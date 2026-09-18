from transcriber import transcribe

segments = transcribe('samples/input/QU1Fk-XzT-A.wav')[:5]
for s in segments:
    print(f"{s['start']:.2f} - {s['end']:.2f}: {s['text']}")