import copy
import statistics
import sys
from collections import defaultdict

import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Iterable, Set

import optuna
import sklearn
from optuna import Trial
from sklearn import metrics
from sklearn.cluster import OPTICS, HDBSCAN
import numpy as np
import numpy.typing as npt

from extractive_approach.entity_clustering import EntityCompatibilityCollection, EntityClustering, EntityCluster
from template_lib.data_classes.Entity import Entity


class ClusteringApproach(ABC):
    entities: List[Entity]
    entity_compatibilities: EntityCompatibilityCollection
    max_slots_cardinalities: Dict[str, int]

    def __init__(self,
                 entities: List[Entity],
                 entity_compatibilities: EntityCompatibilityCollection,
                 template_type: str,
                 max_slots_cardinalities: Optional[Dict[str, int]] = None):
        self.entities = entities
        self.entity_compatibilities = entity_compatibilities
        self.template_type = template_type
        self.max_slots_cardinalities = max_slots_cardinalities if max_slots_cardinalities is not None else dict()
        # Add functional slots of max cardinalities to (maybe empty) functional slots given to compatibility object
        self.entity_compatibilities.functional_slots.update(self.functional_slots)

    @staticmethod
    @abstractmethod
    def param_space(trial: Trial):
        pass

    @abstractmethod
    def clusters(self):
        pass

    @property
    def dist_matrix(self):
        return np.array([
            [0.0 if ent1 == ent2
             else (sys.float_info.max if (ent1.get_referencing_slot_names() == ent2.get_referencing_slot_names() # math.inf throws exceptions for sklearn
                                and ent1.get_referencing_slot_names().issubset(self.functional_slots))
                   else 1.0 - self.entity_compatibilities[(ent1, ent2)])
             for ent2 in self.entities]
            for ent1 in self.entities
        ])

    @property
    def functional_slots(self) -> Set[str]:
        return {k for k, v in self.max_slots_cardinalities.items() if v == 1}

    @property
    def min_clusters(self) -> int:
        functional_entities = defaultdict(list)
        for ent in self.entities:
            if ent.get_referencing_slot_names().issubset(self.functional_slots):
                for slot in ent.get_referencing_slot_names():
                    functional_entities[slot].append(ent)

        return max([1] + [len(val) for val in functional_entities.values()])

    def labels_to_clustering(self, labels: Iterable[int]) -> EntityClustering:
        if -1 in labels:
            print("Warning: Clustering contains unassigned entities (label -1)", labels)

        cluster_ids = defaultdict(list)

        for idx, label in enumerate(labels):
            cluster_ids[label].append(idx)

        print(cluster_ids)

        return EntityClustering([
            EntityCluster({self.entities[idx] for idx in vals})
            for vals in cluster_ids.values()
        ], template_type=self.template_type)


class BeamSearchClustering(ClusteringApproach):

    def __init__(self,
                 entities: List[Entity],
                 entity_compatibilities: EntityCompatibilityCollection,
                 template_type: str,
                 beam_size: int,
                 max_slots_cardinalities: Optional[Dict[str, int]] = None,
                 empty_clusters_weight=0.5,
                 exponential_factor=0.25):
        super().__init__(entities=entities,
                         entity_compatibilities=entity_compatibilities,
                         template_type=template_type,
                         max_slots_cardinalities=max_slots_cardinalities)
        self.beam_size = beam_size
        self.empty_clusters_weight = empty_clusters_weight
        self.exponential_factor = exponential_factor

    @staticmethod
    def param_space(trial: Trial):
        return {
            'beam_size': trial.suggest_int('beam_size', 1, 25),
            'empty_clusters_weight': trial.suggest_float('empty_clusters_weight', 0.0, 1.0),
            'exponential_factor': trial.suggest_float('exponential_factor', 0.00001, 1.0, log=True),
        }

    @staticmethod
    def optimize_entity_clustering(entities: List[Entity],
                                   entity_compatibilities: EntityCompatibilityCollection,
                                   template_type: str,
                                   beam_size: int,
                                   num_clusters: int) -> EntityClustering:
        assert num_clusters <= len(entities)
        clustering_candidates: List[Tuple[frozenset[Entity], EntityClustering]]
        clustering_candidates = [
            (frozenset(entities), EntityClustering.from_num_clusters(num_clusters=num_clusters, template_type=template_type))
            for _ in range(beam_size)
        ]

        for _ in range(len(entities)):
            successor_candidates: List[Tuple[frozenset[Entity], EntityClustering]] = []
            for free_entities, cand in clustering_candidates:
                for ent in free_entities:
                    for cl_idx in range(num_clusters):
                        new_cand = cand.copy()  # copy.deepcopy(cand)
                        new_cand.clusters[cl_idx].add_entity(entity=ent)
                        successor_candidates.append(
                            ((free_entities.difference({ent})), new_cand)
                        )

            clustering_candidates = list(
                sorted(successor_candidates,
                       key=lambda x: x[1].clustering_compatibility(entity_compatibilities),
                       reverse=True)
            )[:beam_size]

        return clustering_candidates[0][1]

    def clusters(self):
        # TODO: Score might be unnecessary, we can just increase cluster number until we get first empty cluster and
        #  after that choose clustering with highest intra-cluster compatibility
        def score(cluster: EntityClustering):
            comp = cluster.clustering_compatibility(entity_compatibilities_collection=self.entity_compatibilities)
            small_clusters = cluster.num_clusters_smaller_than(n=2)
            return (
                    (1 - self.empty_clusters_weight) * comp
                    + self.empty_clusters_weight * math.exp(-self.exponential_factor * (small_clusters - 1))
            ) / 2.0

        clusterings = []

        for n in range(1, len(self.entities) + 1):
            cl = self.optimize_entity_clustering(entities=self.entities,
                                                 entity_compatibilities=self.entity_compatibilities,
                                                 template_type=self.template_type,
                                                 beam_size=self.beam_size,
                                                 num_clusters=n)
            if cl.num_empty_clusters > 0:
                # if cl.num_clusters_smaller_than(2) > 0:
                print("First empty cluster at ", n)
                break
            else:
                clusterings.append(cl)
        print([score(cl) for cl in clusterings])

        return list(sorted(clusterings, key=score, reverse=True))[0]


class BeamSearchOptunaClustering(ClusteringApproach):

    def __init__(self,
                 entities: List[Entity],
                 entity_compatibilities: EntityCompatibilityCollection,
                 template_type: str,
                 num_trials: int,
                 beam_size: int,
                 empty_clusters_weight: float,
                 exponential_factor: float,
                 max_slots_cardinalities: Optional[Dict[str, int]] = None):
        super().__init__(entities=entities,
                         entity_compatibilities=entity_compatibilities,
                         template_type=template_type,
                         max_slots_cardinalities=max_slots_cardinalities)
        self.num_trials = num_trials
        self.beam_size = beam_size
        self.empty_clusters_weight = empty_clusters_weight
        self.exponential_factor = exponential_factor

    @staticmethod
    def param_space(trial: Trial):
        return {
            'num_trials': trial.suggest_int('num_trials', 10, 10),
            'beam_size': trial.suggest_int('beam_size', 20, 20),
            'empty_clusters_weight': trial.suggest_float('empty_clusters_weight', 0.25, 0.5),
            'exponential_factor': trial.suggest_float('exponential_factor', 0.01, 0.1, log=True),
        }

    def clusters(self):
        def objective(trial):
            clustering = BeamSearchClustering.optimize_entity_clustering(
                entities=self.entities,
                entity_compatibilities=self.entity_compatibilities,
                template_type=self.template_type,
                beam_size=self.beam_size,
                num_clusters=trial.suggest_int('num_clusters', 1, len(self.entities))
            )
            comp = clustering.clustering_compatibility(entity_compatibilities_collection=self.entity_compatibilities)
            small_clusters = clustering.num_clusters_smaller_than(n=2)
            score = (
                            (1 - self.empty_clusters_weight) * comp
                            + self.empty_clusters_weight * math.exp(-self.exponential_factor * (small_clusters - 1))
                    ) / 2.0

            print(score)

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.num_trials, n_jobs=-1)
        print(study.best_params)

        return BeamSearchClustering.optimize_entity_clustering(entities=self.entities,
                                                               entity_compatibilities=self.entity_compatibilities,
                                                               template_type=self.template_type,
                                                               beam_size=self.beam_size,
                                                               **study.best_params)


class BeamSearchOptunaMultiClustering(ClusteringApproach):

    def __init__(self,
                 entities: List[Entity],
                 entity_compatibilities: EntityCompatibilityCollection,
                 template_type: str,
                 num_trials: int,
                 beam_size: int,
                 max_slots_cardinalities: Optional[Dict[str, int]] = None):
        super().__init__(entities=entities,
                         entity_compatibilities=entity_compatibilities,
                         template_type=template_type,
                         max_slots_cardinalities=max_slots_cardinalities)
        self.num_trials = num_trials
        self.beam_size = beam_size

    @staticmethod
    def param_space(trial: Trial):
        return {
            'num_trials': trial.suggest_int('num_trials', 10, 10),
            'beam_size': trial.suggest_int('beam_size', 20, 20),
        }

    def clusters(self):
        def objective(trial):
            clustering = BeamSearchClustering.optimize_entity_clustering(
                entities=self.entities,
                entity_compatibilities=self.entity_compatibilities,
                template_type=self.template_type,
                beam_size=self.beam_size,
                num_clusters=trial.suggest_int('num_clusters', 1, len(self.entities))
            )
            comp = clustering.clustering_compatibility(entity_compatibilities_collection=self.entity_compatibilities)
            small_clusters = clustering.num_clusters_smaller_than(n=2)

            return comp, 0.0 if len(clustering.clusters) < 2 else statistics.variance(
                [len(cl) for cl in clustering.clusters]), small_clusters

        study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
        study.optimize(objective, n_trials=self.num_trials, n_jobs=-1)
        # print(study.best_params)

        return BeamSearchClustering.optimize_entity_clustering(entities=self.entities,
                                                               entity_compatibilities=self.entity_compatibilities,
                                                               template_type=self.template_type,
                                                               beam_size=self.beam_size,
                                                               **list(study.best_trials)[0].params)


class OPTICSClustering(ClusteringApproach):

    def __init__(self,
                 entities: List[Entity],
                 entity_compatibilities: EntityCompatibilityCollection,
                 template_type: str,
                 max_slots_cardinalities: Optional[Dict[str, int]] = None,
                 min_samples: int = 2,
                 xi: float = 0.05):
        super().__init__(entities=entities,
                         entity_compatibilities=entity_compatibilities,
                         template_type=template_type,
                         max_slots_cardinalities=max_slots_cardinalities)
        self.min_samples = min_samples
        self.xi = xi

    @staticmethod
    def param_space(trial: Trial):
        return {
            'min_samples': trial.suggest_int('min_samples', 2, 2),
            'xi': trial.suggest_float('xi', 0.0, 1.0),
        }

    def clusters(self):
        c = OPTICS(min_samples=self.min_samples, xi=self.xi, metric="precomputed")
        if self.dist_matrix.shape == (1, 1):
            return self.labels_to_clustering([0])
        else:
            clusters = c.fit(self.dist_matrix)
            return self.labels_to_clustering(clusters.labels_)


class HDBSCANClustering(ClusteringApproach):

    def __init__(self,
                 entities: List[Entity],
                 entity_compatibilities: EntityCompatibilityCollection,
                 template_type: str,
                 max_slots_cardinalities: Optional[Dict[str, int]] = None,
                 min_samples: int = 2):
        super().__init__(entities=entities,
                         entity_compatibilities=entity_compatibilities,
                         template_type=template_type,
                         max_slots_cardinalities=max_slots_cardinalities)
        self.min_samples = min_samples

    @staticmethod
    def param_space(trial: Trial):
        return {
            'min_samples': trial.suggest_int('min_samples', 2, 2)
        }

    def clusters(self):
        c = HDBSCAN(min_samples=self.min_samples, metric="precomputed", min_cluster_size=2, allow_single_cluster=True)
        if self.dist_matrix.shape == (1, 1):
            return self.labels_to_clustering([0])
        else:
            clusters = c.fit(self.dist_matrix)
            return self.labels_to_clustering(clusters.labels_)


class HDBSCANOptunaClustering(ClusteringApproach):

    def __init__(self,
                 entities: List[Entity],
                 entity_compatibilities: EntityCompatibilityCollection,
                 template_type: str,
                 num_trials: int,
                 max_slots_cardinalities: Optional[Dict[str, int]] = None):
        super().__init__(entities=entities,
                         entity_compatibilities=entity_compatibilities,
                         template_type=template_type,
                         max_slots_cardinalities=max_slots_cardinalities)
        self.num_trials = num_trials

    @staticmethod
    def param_space(trial: Trial):
        return {
            'num_trials': trial.suggest_int('num_trials', 5, 20)
        }

    def _run_clustering(self, min_samples: int):
        c = HDBSCAN(min_samples=min_samples, metric="precomputed", min_cluster_size=2, allow_single_cluster=True)
        if self.dist_matrix.shape == (1, 1):
            return self.labels_to_clustering([0])
        else:
            clusters = c.fit(self.dist_matrix)
            return self.labels_to_clustering(clusters.labels_)

    def clusters(self):
        def objective(trial):
            clustering = self._run_clustering(
                min_samples=trial.suggest_int('min_samples', 1, len(self.entities))
            )
            comp = clustering.clustering_compatibility(entity_compatibilities_collection=self.entity_compatibilities)
            small_clusters = clustering.num_clusters_smaller_than(n=2)

            return comp, 0.0 if len(clustering.clusters) < 2 else statistics.variance(
                [len(cl) for cl in clustering.clusters]), small_clusters

        study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
        study.optimize(objective, n_trials=self.num_trials, n_jobs=-1)
        # print(study.best_params)

        return self._run_clustering(**list(study.best_trials)[0].params)


class AgglomerativeClustering(ClusteringApproach):

    def __init__(self,
                 entities: List[Entity],
                 entity_compatibilities: EntityCompatibilityCollection,
                 template_type: str,
                 distance_threshold: float,
                 max_slots_cardinalities: Optional[Dict[str, int]] = None,
                 linkage: str = "average"):
        super().__init__(entities=entities,
                         entity_compatibilities=entity_compatibilities,
                         template_type=template_type,
                         max_slots_cardinalities=max_slots_cardinalities)
        self.distance_threshold = distance_threshold
        self.linkage = linkage

    @staticmethod
    def param_space(trial: Trial):
        return {
            'distance_threshold': trial.suggest_float('distance_threshold', 0.0, 1.0),
            'linkage': trial.suggest_categorical('linkage', ['complete', 'average', 'single']),
        }

    def clusters(self):
        c = sklearn.cluster.AgglomerativeClustering(n_clusters=None,
                                                    distance_threshold=self.distance_threshold,
                                                    linkage=self.linkage,
                                                    metric="precomputed")
        if self.dist_matrix.shape == (1, 1):
            return self.labels_to_clustering([0])
        else:
            clusters = c.fit(self.dist_matrix)
            return self.labels_to_clustering(clusters.labels_)

class DummyClustering(ClusteringApproach):

    def __init__(self,
                 entities: List[Entity],
                 entity_compatibilities: EntityCompatibilityCollection,
                 template_type: str,
                 max_slots_cardinalities: Optional[Dict[str, int]] = None):
        super().__init__(entities=entities,
                         entity_compatibilities=entity_compatibilities,
                         template_type=template_type,
                         max_slots_cardinalities=max_slots_cardinalities)

    @staticmethod
    def param_space(trial: Trial):
        return {
        }

    def clusters(self):
        return self.labels_to_clustering(list(range(len(self.entities))))
