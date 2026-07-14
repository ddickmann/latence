# Storage seam

The fsspec-backed IO layer that makes the framework cloud-agnostic
([ADR-0009](../adr/0009-fsspec-storage-seam.md)) — `file://`, `s3://`, `gcs://`, `az://` are
one line of config apart.

::: latence_core.storage
