"""RustPAM: High-performance PAM clustering with Rust."""
# isort: skip_file
# Import order is critical to avoid circular imports


from .rustpam import swap_eager

# 然后导入 Python 包装类（依赖于 swap_eager）
from .onebatchpam import OneBatchPAM

__version__ = "0.1.0"
__all__ = ["OneBatchPAM", "swap_eager"]
