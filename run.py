import outetts
import time

model_config = outetts.GGUFModelConfig_v1(
    model_path="OuteTTS-0.2-500M-FP16.gguf",
    language="en", # Supported languages in v0.2: en, zh, ja, ko
    n_gpu_layers=-1,
)

interface = outetts.InterfaceGGUF(model_version="0.2", cfg=model_config)

# # Print available default speakers
# interface.print_default_speakers()

# # Load a default speaker
speaker = interface.load_default_speaker(name="male_1")

time_start = time.time()
# Generate speech
output = interface.generate(
    text="Nexa AI is a Cupertino-based company founded in April 2023.",
    temperature=0.1,
    repetition_penalty=1.1,
    max_length=4096,

    # Optional: Use a speaker profile for consistent voice characteristics
    # Without a speaker profile, the model will generate a voice with random characteristics
    # speaker=speaker,
)

# Save the generated speech to a file
output.save("output.wav")

# Optional: Play the generated audio
# output.play()

time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")