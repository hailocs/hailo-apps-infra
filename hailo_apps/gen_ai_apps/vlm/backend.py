import time
import multiprocessing as mp
import numpy as np
import cv2
from hailo_platform import VDevice
from hailo_platform.genai import VLM

def vlm_worker_process(request_queue, response_queue, hef_path, max_tokens, temperature, seed):
    # Standalone worker function that creates its own VLM instance
    try:
        vdevice = VDevice()
        vlm = VLM(vdevice, hef_path)
        
        while True:
            item = request_queue.get()
            if item is None:
                break
            try:
                # Use custom max_tokens if provided, otherwise use default
                current_max_tokens = item.get('max_tokens', max_tokens)
                result = _hailo_inference_inner(item['numpy_image'], vlm, item['trigger'], item['prompts'], current_max_tokens, temperature, seed)
                response_queue.put({'result': result, 'error': None})
            except Exception as e:
                import traceback
                response_queue.put({'result': None, 'error': f"{str(e)}\n{traceback.format_exc()}"})
    
    except Exception as e:
        response_queue.put({'result': None, 'error': f"Worker initialization failed: {str(e)}"})
    finally:
        try:
            vlm.release()
            vdevice.release()
        except:
            pass

def _create_structured_prompt(prompts, trigger):
    """Create structured prompt based on trigger type"""
    # Get the user prompt text based on trigger type
    if trigger == 'custom':
        user_text = prompts['hailo_user_prompt']
    else:
        user_text = prompts['hailo_user_prompt'].replace(
            "{details}", 
            prompts['use_cases'][trigger]["details"]
        )
    
    return [
        {
            "role": "system", 
            "content": [{"type": "text", "text": prompts['hailo_system_prompt']}]
        },
        {
            "role": "user", 
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": user_text}
            ]
        }
    ]

def _hailo_inference_inner(image, vlm, trigger, prompts, max_tokens, temperature, seed):
    """Standalone inference function"""
    structured_prompt = _create_structured_prompt(prompts, trigger)
    
    try:
        response = ''
        start_time = time.time()
        with vlm.generate(prompt=structured_prompt, frames=[image], temperature=temperature, seed=seed, max_generated_tokens=max_tokens) as generation:
            num_tokens = 0
            for chunk in generation:
                if trigger == 'custom':
                    if chunk != '<|im_end|>':
                        print(chunk, end='', flush=True)
                response += chunk
                num_tokens += 1
                if num_tokens > max_tokens:
                    vlm.clear_context()
                    return {'answer': 'Too many tokens in response', 'time': f"{time.time() - start_time:.2f} seconds"}
        vlm.clear_context()
        
        # Parse response
        response = response.replace('<|im_end|>', '').strip()
        
        if trigger == 'custom':
            # For custom prompts, return the full response
            parsed_response = response
        else:
            # For predefined use cases, parse for specific options
            for option in prompts['use_cases'][trigger]['options']:
                if option in response:
                    parsed_response = option
                    break
            else:
                parsed_response = "No Event Detected"
            
        return {'answer': parsed_response, 'time': f"{time.time() - start_time:.2f} seconds"}
    except Exception as e:
        return {'answer': f'Error: {str(e)}', 'time': f"{time.time() - start_time:.2f} seconds"}

class Backend:
    def __init__(self, prompts=None):
        self.prompts = prompts
        self.hef_path = 'Qwen2-VL-2B-Instruct.hef'
        self.max_tokens = 40
        self.temperature = 0.1
        self.seed = 42
        self.trigger = list(self.prompts['use_cases'].keys())[0]
        self._request_queue = mp.Queue()
        self._response_queue = mp.Queue()
        self._process = mp.Process(target=vlm_worker_process, args=(self._request_queue, self._response_queue, self.hef_path, self.max_tokens, self.temperature, self.seed))
        self._process.start()

    def _clear_queues(self):
        """Clear both request and response queues"""
        while not self._request_queue.empty():
            self._request_queue.get_nowait()
        while not self._response_queue.empty():
            self._response_queue.get_nowait()

    def _execute_inference(self, request_data, timeout, error_prefix):
        """Common inference execution logic"""
        self._request_queue.put(request_data)
        try:
            response = self._response_queue.get(timeout=timeout)
        except Exception:
            self._clear_queues()
            return {'answer': f'{error_prefix} timeout', 'time': f'{timeout} seconds'}
        
        if response['error']:
            return {'answer': f'{error_prefix} error', 'time': '30 seconds'}
        return response['result']

    def hailo_inference(self, image):
        request_data = {
            'numpy_image': self.convert_resize_image(image), 
            'trigger': self.trigger, 
            'prompts': self.prompts
        }
        return self._execute_inference(request_data, timeout=20, error_prefix='Hailo')

    def vlm_custom_inference(self, image, custom_prompt):
        """Run VLM inference with a custom user prompt"""
        custom_prompts = {
            'hailo_system_prompt': "You are a helpful assistant that analyzes images and answers questions about them.",
            'hailo_user_prompt': custom_prompt,
            'use_cases': {'custom': {'details': custom_prompt, 'options': []}}
        }
        request_data = {
            'numpy_image': self.convert_resize_image(image), 
            'trigger': 'custom', 
            'prompts': custom_prompts,
            'max_tokens': 200  # Use higher token limit for custom questions
        }
        return self._execute_inference(request_data, timeout=30, error_prefix='VLM')

    def close(self):
        """Clean shutdown of the backend"""
        try:
            self._request_queue.put(None)  # Signal worker to stop
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()
        except Exception as e:
            print(f"Error closing backend: {e}")

    @staticmethod
    def convert_resize_image(image_array, target_shape=[336, 336, 3], target_dtype=np.uint8):
        target_height, target_width, _ = target_shape
        img = image_array.copy()
        if len(img.shape) == 3 and img.shape[2] == 3:  # Convert BGR to RGB if it's a 3-channel image (OpenCV uses BGR by default)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != target_height or img.shape[1] != target_width:  # Resize if needed
            img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        img = img.astype(target_dtype)
        return img