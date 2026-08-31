"""Optional scientific and compute plugins for Molly Core v2.

The package is intentionally not imported by :mod:`molly.core`.  Applications
opt in to a plugin by registering its host-owned ToolSpecs explicitly.
"""

__all__ = ["br1_inverse_design", "remote_compute"]
