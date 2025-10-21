#!/bin/bash

# Script to run Hailo8L models on Hailo 8 tests
# This script runs all the tests that test Hailo8L models on Hailo 8 architecture

echo "Running Hailo8L models on Hailo 8 tests..."
echo "=========================================="

# Check if we're on Hailo 8 architecture
echo "Checking Hailo architecture..."
python3 -c "
from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.defines import HAILO8_ARCH
arch = detect_hailo_arch()
print(f'Detected Hailo architecture: {arch}')
if arch != HAILO8_ARCH:
    print('WARNING: Not running on Hailo 8 architecture. Some tests will be skipped.')
else:
    print('Running on Hailo 8 architecture. All tests will be executed.')
"

echo ""
echo "Running individual pipeline tests..."
echo "==================================="

# Run tests from test_all_pipelines.py
echo "1. Running Hailo8L models on Hailo 8 from test_all_pipelines.py..."
pytest tests/test_all_pipelines.py::test_hailo8l_models_on_hailo8 -v
pytest tests/test_all_pipelines.py::test_hailo8l_models_on_hailo8_comprehensive -v

echo ""
echo "2. Running Hailo8L models on Hailo 8 from test_multisource.py..."
pytest tests/test_multisource.py::test_hailo8l_models_on_hailo8_multisource -v

echo ""
echo "3. Running Hailo8L models on Hailo 8 from test_reid.py..."
pytest tests/test_reid.py::test_hailo8l_models_on_hailo8_reid -v

echo ""
echo "4. Running Hailo8L models on Hailo 8 from test_face_recon.py..."
pytest tests/test_face_recon.py::test_hailo8l_models_on_hailo8_face_recon -v

echo ""
echo "5. Running comprehensive Hailo8L on Hailo 8 tests..."
pytest tests/test_hailo8l_on_hailo8_comprehensive.py -v

echo ""
echo "All Hailo8L on Hailo 8 tests completed!"
echo "Check the logs in the 'logs/' directory for detailed results."
