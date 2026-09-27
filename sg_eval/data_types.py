from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Object3D:
    """Represents a 3D object with label and bounding box."""
    id: int
    label: str
    position: np.ndarray  # [x, y, z]
    size: np.ndarray      # [sx, sy, sz]
    angle_z: float = 0.0  # yaw angle in radians (rotation around Z axis)

    @classmethod
    def from_dict(cls, data: dict) -> "Object3D":
        return cls(
            id=int(data["id"]),
            label=str(data["label"]).lower().strip(),
            position=np.array(data["position"], dtype=np.float32),
            size=np.array(data["size"], dtype=np.float32),
            angle_z=float(data.get("angle_z", 0.0)),
        )


@dataclass
class Relationship:
    """Represents a scene graph relationship triplet."""
    subject_id: int
    object_id: int
    predicate: str

    @classmethod
    def from_list(cls, data: list) -> "Relationship":
        return cls(
            subject_id=int(data[0]),
            object_id=int(data[1]),
            predicate=str(data[2]).lower().strip(),
        )


@dataclass
class SceneGraph:
    """A full scene graph with objects and relationships."""
    objects: List[Object3D] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneGraph":
        sg = cls()
        for obj_data in data.get("objects", []):
            sg.objects.append(Object3D.from_dict(obj_data))
        for rel_data in data.get("relationships", []):
            sg.relationships.append(Relationship.from_list(rel_data))
        return sg
