from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any

from instantonanalysis.instanton._typing import TYPE_CHECKING
from instantonanalysis.instanton.schemas.xconfig import XConfig

if TYPE_CHECKING:
    from instantonanalysis.instanton._typing import xrArray
    

class IBox(ABC):
    """Common interface for spatial bounding boxes/selectors."""

    _possible_names: dict[str, List[str]] = {}
    xconfig: Optional[XConfig] = None

    @abstractmethod
    def _check_bounds(self, ds: xrArray, dims: Tuple[str, ...]) -> None:
        raise NotImplementedError

    def _find_coord_names(self, ds: xrArray) -> Tuple[str, ...]:
        """Discover coordinate names using the _possible_names map."""
        found_coords = set(ds.coords) | set(ds.dims)
        names = []
        
        for category, candidates in self._possible_names.items():
            name = next((c for c in candidates if c in found_coords), None)
            if not name:
                raise ValueError(
                    f"Required {category} coordinate not found. "
                    f"Expected one of {candidates}. Found: {list(ds.coords)}"
                )
            names.append(name)
        
        return tuple(names)

    @abstractmethod
    def enforce_coords(self, ds: xrArray, xconfig: Optional[XConfig] = None) -> xrArray:
        raise NotImplementedError

    def extract(
            self,
            ds: xrArray,
            xconfig: Optional[Any] = None,
            dims: Optional[Tuple[str, ...]] = None,
        ) -> xrArray:
        res = self.enforce_coords(ds, xconfig)
        _dims = self.get_names(res, xconfig)

        self._check_bounds(res, _dims)
        try:
            subset = self.select(res, _dims)
            if any(subset.sizes[d] == 0 for d in dims):
                raise ValueError(f"Extraction resulted in an empty dataset for box: {self}")
            return subset
        except Exception as e:
            raise RuntimeError(f"Error during spatial extraction: {str(e)}")

    def get_names(self, ds: xrArray, xconfig: Optional[XConfig] = None) -> Tuple[str, ...]:
        """Generalized: Uses xconfig if present, otherwise discovers names."""
        if xconfig:
            return xconfig.spatial_dims
        if self.xconfig:
            return self.xconfig.spatial_dims
        return self._find_coord_names(ds)

    @abstractmethod
    def select(self, series: xr.DataArray, dims: Tuple[str, ...]) -> xr.DataArray:
        """Perform the actual coordinate selection logic."""
        raise NotImplementedError




