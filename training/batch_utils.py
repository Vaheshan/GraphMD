from dataclasses import dataclass
from typing import Dict, Any, List

from torch_geometric.data import Data, Batch


@dataclass
class GraphBatch:
    """
    Container bundling batched protein and pocket graphs.

    Attributes:
        protein: torch_geometric.data.Batch over residue graphs.
        pocket: torch_geometric.data.Batch over pocket atom graphs.
        labels: Optional dictionary for task-specific labels (e.g., stability, affinity).
    """

    protein: Batch
    pocket: Batch
    labels: Dict[str, Any]


def collate_complexes(
    protein_graphs: List[Data],
    pocket_graphs: List[Data],
    labels: Dict[str, Any],
) -> GraphBatch:
    """
    Collate lists of per-complex protein and pocket graphs into batched graphs.

    This is a lightweight helper around torch_geometric.data.Batch.from_data_list
    that also packages associated labels.

    Args:
        protein_graphs: List of residue-level Data objects.
        pocket_graphs: List of pocket atom Data objects.
        labels: Dictionary of label tensors keyed by task name. Each tensor
            should have a leading batch dimension matching len(protein_graphs).

    Returns:
        GraphBatch with batched protein/pocket graphs and labels.
    """
    if len(protein_graphs) != len(pocket_graphs):
        raise ValueError(
            f"protein_graphs and pocket_graphs must have the same length, "
            f"got {len(protein_graphs)} and {len(pocket_graphs)}"
        )

    protein_batch = Batch.from_data_list(protein_graphs)
    pocket_batch = Batch.from_data_list(pocket_graphs)
    return GraphBatch(protein=protein_batch, pocket=pocket_batch, labels=labels)

