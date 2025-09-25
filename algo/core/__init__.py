"""
Core Algorithm Modules

This package contains the generalized, reusable components for sector-based
quantitative analysis across all GICS sectors.
"""

__version__ = "1.0.0"

# Import main components for easy access
try:
    from .config import (
        SECTOR_GICS_MAPPING, 
        get_sector_config, 
        set_sector_config
    )
    from .pipeline import run
    
    __all__ = [
        'SECTOR_GICS_MAPPING',
        'get_sector_config', 
        'set_sector_config',
        'run'
    ]
except ImportError:
    # Handle case where dependencies aren't available
    __all__ = []
