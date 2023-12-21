from typing import Tuple, Set, Iterable, List, Optional
from template_lib.data_classes.Entity import Entity
from template_lib.data_classes.Template import Template
import itertools


class EntityCompatibilityCollection:
    def __init__(self, functional_slots: Optional[Set[str]] = None):
        if functional_slots is None:
            functional_slots = set()
        self.functional_slots = functional_slots
        self._entity_compatibilities = dict()


    def add_entity_compatibility(self, entity_pair: Tuple[Entity, Entity], compatibility: float):
        entity_pair_set = frozenset(entity_pair)
        #assert len(entity_pair_set) == 2
        #rev_entity_pair_set = frozenset((entity_pair[1], entity_pair[0]))

        #if entity_pair_set in self._entity_compatibilities:# or rev_entity_pair_set in self._entity_compatibilities:
        #    raise KeyError('entity pair already existing')

        self._entity_compatibilities[entity_pair_set] = compatibility
        #self._entity_compatibilities[rev_entity_pair_set] = compatibility

    def __getitem__(self, entity_pair: Tuple[Entity, Entity]):
        if entity_pair[0] == entity_pair[1]:
            return 1.0
        #ent1.get_referencing_slot_names() == ent2.get_referencing_slot_names() and ent1.get_referencing_slot_names().issubset(self.functional_slots)
        elif (entity_pair[0].get_referencing_slot_names() == entity_pair[1].get_referencing_slot_names()
              and entity_pair[0].get_referencing_slot_names().issubset(self.functional_slots)):
            return 0.0
        else:
            entity_pair_set = frozenset(entity_pair)
            if entity_pair_set not in self._entity_compatibilities:
                raise KeyError('entity pair not in compatibility collection')
            else:
                return self._entity_compatibilities[entity_pair_set]


class EntityCluster:
    def __init__(self, entities=None):
        if entities is None:
            entities = set()
        self.entities: Set[Entity] = entities

    def add_entity(self, entity: Entity):
        if entity in self.entities:
            raise KeyError('entity already existing in cluster')
        else:
            self.entities.add(entity)

    def cluster_compatibility(self, entity_compatibilities_collection: EntityCompatibilityCollection):
        # if there are no entity pairs, return None/neutral score
        if len(self.entities) < 2:
            return 0.5#TODO: None

        entity_pairs = itertools.combinations(self.entities, 2)
        compatibility_scores = [
            entity_compatibilities_collection[tuple(entity_pair)] for entity_pair in entity_pairs
        ]

        # return mean compatibility score
        return sum(compatibility_scores) / len(compatibility_scores)

    def to_template(self, templayte_type: str, template_id: str):
        template = Template(templayte_type, template_id)

        # assign entities to template
        for entity in self.entities:
            assert entity.get_label() is not None
            template.add_slot_filler(entity.get_label(), entity)

        return template

    def __len__(self):
        return len(self.entities)

    def __contains__(self, item):
        return item in self.entities

    def copy(self):
        return EntityCluster({entity for entity in self.entities})

    def __str__(self):
        return str(self.entities)


class EntityClustering:
    def __init__(self, clusters: Iterable[EntityCluster], template_type: str):
        self.clusters: List[EntityCluster] = list(clusters)
        self.template_type = template_type

    @classmethod
    def from_num_clusters(cls, num_clusters: int, template_type: str):
        return cls(clusters=[EntityCluster() for _ in range(num_clusters)], template_type=template_type)

    def clustering_compatibility(self, entity_compatibilities_collection: EntityCompatibilityCollection):
        scores = [
            cl.cluster_compatibility(entity_compatibilities_collection) for cl in self.clusters
            if cl.cluster_compatibility(entity_compatibilities_collection) is not None
        ]

        if len(scores) == 0:
            return 0.0
        else:
            return sum(scores) / len(scores)

    def labels(self, entities: List[Entity]):
        label_map = {ent: i for i in range(len(self.clusters)) for ent in self.clusters[i].entities}
        return [label_map[entity] for entity in entities]

    @property
    def templates(self):
        return [
            cl.to_template(templayte_type=self.template_type, template_id=f"{self.template_type}_{num}")
            for num, cl in enumerate(self.clusters)
        ]

    def clusters_smaller_than(self, n):
        return [cl for cl in self.clusters if len(cl) < n]

    def num_clusters_smaller_than(self, n):
        return len(self.clusters_smaller_than(n=n))

    @property
    def num_empty_clusters(self):
        return self.num_clusters_smaller_than(n=1)

    def copy(self):
        return EntityClustering([cluster.copy() for cluster in self.clusters], template_type=self.template_type)

    def __str__(self):
        return "\n".join([str((str(idx), str(cl))) for idx, cl in enumerate(self.clusters)])





