import torch
from torch_geometric.data import Data

from cell_graphs.data.structures.graph_bag import GraphBag, GraphBagBatch


def _generate_graph() -> Data:
    """Generate a graph with random features and number of nodes."""
    num_nodes = torch.randint(5, 11, (1,)).item()
    return Data(
        x=torch.rand((num_nodes, 3)),
        y=torch.rand(num_nodes, 1),
        edge_index=torch.randn((2, num_nodes)),
        id=torch.randint(0, 10000, (1, 1)),
    )


def _generate_graph_bags() -> list[GraphBag]:
    """Generate a bag of randomly generated graphs."""
    graph_bags = []
    for _ in range(10):
        num_graphs = torch.randint(1, 5, (1,)).item()
        graphs = [_generate_graph() for _ in range(num_graphs)]
        graph_bag = GraphBag.from_data_list(graphs)
        graph_bag.add_padding(4)
        graph_bags.append(graph_bag)
    return graph_bags


def _get_bag_ids(bag: GraphBag) -> torch.Tensor:
    bag_len = max([int(key[-1]) for key in bag.keys() if key[-1].isdigit()]) + 1  # noqa: SIM118
    return torch.cat([getattr(bag, f"id_{i}") for i in range(bag_len)])


def test_graph_bag() -> None:
    """Test grouping and ungrouping graphs in a bag."""
    num_graphs = torch.randint(1, 5, (1,)).item()
    graphs = [_generate_graph() for _ in range(num_graphs)]

    bag = GraphBag.from_data_list(graphs)

    bag_graphs = bag.to_data_list()

    for graph, bag_graph in zip(graphs, bag_graphs):
        assert graph.to_dict() == bag_graph.to_dict()


def test_graph_bag_batch() -> None:
    """Test batching and unbatching a bags of graphs."""
    graph_bags = _generate_graph_bags()

    bag_ids = [_get_bag_ids(graph_bag) for graph_bag in graph_bags]

    graph_bag_batch = GraphBagBatch.from_graph_bags(graph_bags)

    ids = torch.cat([graph.id for graph in graph_bag_batch.to_data_list()])

    assert torch.all(torch.cat(bag_ids) == ids)
    assert torch.all(graph_bag_batch.id == ids)

    split_ids = graph_bag_batch.from_tensor(graph_bag_batch.id)

    for split_id, bag_id in zip(split_ids, bag_ids):
        assert torch.all(split_id == bag_id)
