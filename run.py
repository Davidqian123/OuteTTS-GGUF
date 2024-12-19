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
# speaker = interface.load_default_speaker(name="male_1")

time_start = time.time()
# Generate speech
output = interface.generate(
    text="Nexa AI is a Cupertino-based company founded in April 2023, specializing in the research and development of multimodal models and developer tools for on-device AI. Founded by Alex (Stanford PhD) and Zack (Ex-Googler & Stanford MS), the company is best known for its Octopus-series models, which offer capabilities comparable to large-scale language models, including function-calling, multimodality, and action planning, while maintaining efficiency for edge device deployment.",
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