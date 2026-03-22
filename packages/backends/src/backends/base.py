"""Abstract backend interface for Ibis connections."""

from abc import ABC, abstractmethod

import ibis
import ibis.expr.types as ir


class Backend(ABC):
    """Abstract backend interface for Ibis connections."""

    @abstractmethod
    def connect(self) -> ibis.BaseBackend:
        """Create and return an Ibis backend connection."""
        ...

    @abstractmethod
    def disconnect(self, conn: ibis.BaseBackend) -> None:
        """Close the connection."""
        ...

    @abstractmethod
    def load_reference_table(
        self,
        conn: ibis.BaseBackend,
        table_name: str,
        parquet_path: str,
    ) -> ir.Table:
        """Load a reference table from a Parquet file into the backend."""
        ...
