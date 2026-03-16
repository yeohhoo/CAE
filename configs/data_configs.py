from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    'flowers_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['flowers_train'],
        'train_target_root': dataset_paths['flowers_train'],
        'test_source_root': dataset_paths['flowers_test'],
        'test_target_root': dataset_paths['flowers_test'],
    },
    'vggfaces_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['vgg_faces_train'],
        'train_target_root': dataset_paths['vgg_faces_train'],
        'test_source_root': dataset_paths['vgg_faces_test'],
        'test_target_root': dataset_paths['vgg_faces_test'],
    },
    'animalfaces_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['animal_faces_train'],
        'train_target_root': dataset_paths['animal_faces_train'],
        'test_source_root': dataset_paths['animal_faces_test'],
        'test_target_root': dataset_paths['animal_faces_test'],
    },
    
}
