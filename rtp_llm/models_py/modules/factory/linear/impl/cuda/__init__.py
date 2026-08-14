"""CUDA Linear implementations and registration"""

import logging

logger = logging.getLogger(__name__)
logger.debug("Registered CUDA Linear strategies")


from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.utils.arch import is_cuda, get_sm

# Register CUDA strategies
from .f16_linear import CudaF16Linear
from .int8_per_channel_linear import CudaInt8PerChannelLinear

LinearFactory.register(CudaF16Linear)
# Per-channel INT8 (W8A8) is device-agnostic weight dequant + matmul, so it is
# registered unconditionally alongside the f16 fallback.
LinearFactory.register(CudaInt8PerChannelLinear)

if is_cuda():
    from .fp8_deepgemm_linear import CudaFp8DeepGEMMLinear
    from .fp8_per_tensor_linear import CudaFp8PerTensorLinear
    major, minor = get_sm()
    if major >= 10:
        from .fp4_linear import CudaFp4GEMMLinear
        LinearFactory.register(CudaFp4GEMMLinear)
    
    LinearFactory.register(CudaFp8PerTensorLinear)
    LinearFactory.register(CudaFp8DeepGEMMLinear)
