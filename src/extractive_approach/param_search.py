import random
import statistics
from typing import List

import optuna
from optuna import Trial
from optuna.storages import JournalFileStorage, JournalStorage
from sklearn import metrics

from extractive_approach.clustering_approaches import ClusteringApproach, BeamSearchClustering, OPTICSClustering, \
    HDBSCANClustering, AgglomerativeClustering, BeamSearchOptunaClustering, HDBSCANOptunaClustering, \
    BeamSearchOptunaMultiClustering
from template_lib.data_classes.Template import Template
from tests.tests_entity_clustering import get_dataset, gen_comp_entities_labels

def gen_single_run():
    noise_level = random.uniform(0.1, 0.5)  # 0.1

    types = set()

    for document, template_collection in train_dataset:
        types.update(template_collection.get_template_types())

    sample: List[Template]
    sample_size: int

    while True:
        chosen_type = random.choice(list(types))
        instances: List[Template] = []

        for document, template_collection in train_dataset:
            instances += template_collection.get_templates_by_type(chosen_type)

        if len(instances) < 2:
            continue

        sample_size = random.randrange(1, min(4, len(instances)))

        sample = random.sample(instances, sample_size)  # publications[:3]

        compatibility, entities, labels = gen_comp_entities_labels(sample, noise_level=noise_level)
        if len(sample) >= 3 and len(entities) >= 3:
            break

    return compatibility, entities, labels, (chosen_type, sample_size, noise_level)


if __name__ == "__main__":
    train_dataset = get_dataset()

    num_runs = 20

    trial_runs = [gen_single_run() for i in range(num_runs)]
    print([metadata for compatibility, entities, labels, metadata in trial_runs])

    def objective(trial: Trial):
        clustering = trial.suggest_categorical('clustering',
                                               ['BeamOptuna', 'BeamOptunaMulti', 'OPTICS', 'HDBSCAN', 'HDBSCANOptuna', 'Agglomerative'])  # , 'Beam'
        cl_class: type[ClusteringApproach]
        match clustering:
            case 'Beam':
                cl_class = BeamSearchClustering
            case 'BeamOptuna':
                cl_class = BeamSearchOptunaClustering
            case 'BeamOptunaMulti':
                cl_class = BeamSearchOptunaMultiClustering
            case 'OPTICS':
                cl_class = OPTICSClustering
            case 'HDBSCAN':
                cl_class = HDBSCANClustering
            case 'HDBSCANOptuna':
                cl_class = HDBSCANOptunaClustering
            case 'Agglomerative':
                cl_class = AgglomerativeClustering
            case _:
                raise RuntimeError("Unknown clustering method: ", clustering)

        params = cl_class.param_space(trial)

        scores = []
        for i in range(num_runs):
            compatibility, entities, labels, (chosen_type, sample_size, noise_level) = trial_runs[i]

            #print(trial.number, trial.params, chosen_type, sample_size, noise_level, flush=True)
            # res = optimize_entity_clustering(entities=entities, entity_compatibilities=compatibility, num_clusters=3, beam_size=5)
            clustering = cl_class(entities=entities, entity_compatibilities=compatibility, **params)
            res = clustering.clusters()
            scores.append(metrics.adjusted_rand_score(res.labels(entities), labels))
            print(metrics.adjusted_rand_score(res.labels(entities), labels))
            trial.report(statistics.mean(scores), i)
        return statistics.mean(scores)


    storage = JournalStorage(JournalFileStorage("optuna.log"))
    study = optuna.create_study(direction='maximize', study_name="Optuna", storage=storage, load_if_exists=True, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=36, n_jobs=-1)
    print(study.best_params)
