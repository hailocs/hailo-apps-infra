# region imports
# Standard library imports
import os
import logging

# Third-party imports
import pytest

# Local application-specific imports
from hailo_apps.hailo_app_python.core.common.test_utils import (
    run_pipeline_module_with_args, 
    run_pipeline_pythonpath_with_args, 
    run_pipeline_cli_with_args, 
    get_pipeline_args
)
# endregion imports

# Configure logging as needed.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_run_everything')
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Define pipeline configurations.
@pytest.fixture
def pipeline():
    return {
        "name": "tiling",
        "module": "hailo_apps.hailo_app_python.apps.tiling.tiling_pipeline",
        "script": "hailo_apps/hailo_app_python/apps/tiling/tiling_pipeline.py",
        "cli": "hailo-tiling"
    }

# Map each run method label to its corresponding function.
run_methods = {
    'module': run_pipeline_module_with_args,
    'pythonpath': run_pipeline_pythonpath_with_args,
    'cli': run_pipeline_cli_with_args
}

def run_test(pipeline, run_method_name, test_name, args):
    """
    Helper function to run the test logic.
    """
    log_file_path = os.path.join(log_dir, f"{pipeline['name']}_{test_name}_{run_method_name}.log")
    
    if run_method_name == 'module':
        stdout, stderr = run_methods[run_method_name](pipeline['module'], args, log_file_path)
    elif run_method_name == 'pythonpath':
        stdout, stderr = run_methods[run_method_name](pipeline['script'], args, log_file_path)
    elif run_method_name == 'cli':
        stdout, stderr = run_methods[run_method_name](pipeline['cli'], args, log_file_path)
    else:
        pytest.fail(f"Unknown run method: {run_method_name}")
    
    out_str = stdout.decode().lower() if stdout else ""
    err_str = stderr.decode().lower() if stderr else ""
    print(f"Completed: {test_name}, {pipeline['name']}, {run_method_name}: {out_str}")
    assert 'error' not in err_str, f"{pipeline['name']} ({run_method_name}) reported an error in {test_name}: {err_str}"
    assert 'traceback' not in err_str, f"{pipeline['name']} ({run_method_name}) traceback in {test_name} : {err_str}"


@pytest.mark.parametrize('run_method_name', list(run_methods.keys()))
def test_train_singlescale(pipeline, run_method_name):
    test_name = 'test_singlescale_tiling'
    args = get_pipeline_args(suite='single_scaling')
    run_test(pipeline, run_method_name, test_name, args)


@pytest.mark.parametrize('run_method_name', list(run_methods.keys()))
def test_train_multiscale(pipeline, run_method_name):
    test_name = 'test_multiscale_tiling'
    args = []
    run_test(pipeline, run_method_name, test_name, args)

if __name__ == "__main__":
    pytest.main(["-v", __file__])