import random
from typing import List

from gama.genetic_programming.components import (
    Primitive,
    Terminal,
    PrimitiveNode,
    DATA_TERMINAL,
)


def random_terminals_for_primitive(
    primitive_set: dict, primitive: Primitive
) -> List[Terminal]:
    """Return a list with a random Terminal for each required input to Primitive."""
    return [random.choice(primitive_set[term_type]) for term_type in primitive.input]


def random_primitive_node(
    output_type: str, primitive_set: dict, exclude: Primitive = None, exclude_list: List[Primitive] = None
) -> PrimitiveNode:
    """Create a PrimitiveNode with specified output_type and random terminals."""
    if exclude_list is None:
        exclude_list = []
    if exclude is not None:
        exclude_list = exclude_list + [exclude]
    available = [p for p in primitive_set[output_type] if p not in exclude_list]
    if not available:
        raise ValueError(f"No available primitives of type {output_type} after exclusions.")
    primitive = random.choice(available)
    terminals = random_terminals_for_primitive(primitive_set, primitive)
    return PrimitiveNode(primitive, data_node=DATA_TERMINAL, terminals=terminals)


def create_random_expression(
    primitive_set: dict, min_length: int = 1, max_length: int = 3, unique_primitives: bool = False
) -> PrimitiveNode:
    """Create at least min_length and at most max_length chained PrimitiveNodes."""
    individual_length = random.randint(min_length, max_length)
    used_identifiers = []
    
    learner_node = random_primitive_node(
        output_type="prediction", primitive_set=primitive_set, exclude_list=None
    )
    if unique_primitives:
        used_identifiers.append(learner_node._primitive.identifier)
    
    last_primitive_node = learner_node
    for _ in range(individual_length - 1):
        exclude_list = None
        if unique_primitives:
            exclude_list = [p for p in primitive_set[DATA_TERMINAL] if p.identifier in used_identifiers]
        
        try:
            primitive_node = random_primitive_node(
                output_type=DATA_TERMINAL, primitive_set=primitive_set, exclude_list=exclude_list
            )
        except ValueError:
            break
        
        if unique_primitives:
            used_identifiers.append(primitive_node._primitive.identifier)
        last_primitive_node._data_node = primitive_node
        last_primitive_node = primitive_node

    return learner_node
