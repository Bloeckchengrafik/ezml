import dataclasses


@dataclasses.dataclass
class DataDeclaration:
    data: callable
    test_data: callable
    steps_per_epoch: int
    validation_steps: int
