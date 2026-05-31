from __future__ import annotations

import warnings
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

    def _validate_dims(self, dims: Tuple[str, ...]) -> None:
        expected_categories = list(self._possible_names.keys())
        for idx, (dim, expected_cat) in enumerate(zip(dims, expected_categories)):
            expected_aliases = self._possible_names[expected_cat]
            if dim in expected_aliases:
                continue  # correct slot

            # Check if the name belongs to a different category
            actual_cat = None
            for cat, aliases in self._possible_names.items():
                if dim in aliases:
                    actual_cat = cat
                    break

            if actual_cat is not None:
                warnings.warn(
                    f"Dimension '{dim}' at position {idx} is expected to be a "
                    f"{expected_cat} name (one of {expected_aliases}), but it "
                    f"matches the {actual_cat} category. The dims tuple may be "
                    f"in the wrong order. Got dims={dims}.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"Dimension '{dim}' at position {idx} is not a recognised "
                    f"name for any spatial category. Expected a {expected_cat} "
                    f"name (one of {expected_aliases}). Got dims={dims}.",
                    UserWarning,
                    stacklevel=2,
                )

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

        if dims is None:
            dims = _dims

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




