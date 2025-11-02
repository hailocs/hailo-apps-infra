from hailo_platform import VDevice
from hailo_platform.genai import LLM

vdevice = VDevice()
llm = LLM(vdevice, 'Qwen2.5-1.5B-Instruct.hef')
prompt = [
    {"role": "system", "content": [{"type": "text", "text": 'You are a helpful assistant.'}]},
    {"role": "user", "content": [{"type": "text", "text": 'Tell a short joke.'}]}]
print(llm.generate_all(prompt=prompt, temperature=0.1, seed=42, max_generated_tokens=200).split(". [{'type'")[0])
llm.clear_context()
llm.release()
vdevice.release()