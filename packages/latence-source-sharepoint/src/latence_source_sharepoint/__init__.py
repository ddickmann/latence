"""latence-source-sharepoint: the SharePoint/OneDrive Source Provider (#182, ADR-0014/0018).

Exposes :class:`~latence_source_sharepoint.provider.SharePointSource`, registered under the
``latence.providers`` entry-point group as ``source.sharepoint``. Installing this package makes
a SharePoint document library a valid ``source`` Stage for any Pipeline, with no change to core
and no change to any Stage contract — the Source seam is what makes a real connector a plugin
rather than a fork.

The package has **no third-party dependency**: it speaks Microsoft Graph's REST/JSON API over
the Python standard library, so ``latence-core`` takes on nothing new (ADR-0016) and a package
whose job is to hold credentials adds no supply-chain surface.
"""

from __future__ import annotations

from latence_source_sharepoint.graph import GraphClient, GraphItem
from latence_source_sharepoint.provider import SharePointSource
from latence_source_sharepoint.transport import HttpResponse, HttpTransport, UrllibTransport

__all__ = [
    "GraphClient",
    "GraphItem",
    "HttpResponse",
    "HttpTransport",
    "SharePointSource",
    "UrllibTransport",
]
