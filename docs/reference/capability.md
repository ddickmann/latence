# Capability protocols

The narrow, `runtime_checkable` `typing.Protocol`s a Provider implements
([ADR-0004](../adr/0004-capability-protocols-with-provider-plugins.md)) — no base class or
import of core required — plus the `ProviderRegistry` that resolves Providers by name.

::: latence_core.capability
