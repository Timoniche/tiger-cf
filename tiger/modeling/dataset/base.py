import json
import logging

from modeling.dataset.samplers import TrainSampler, EvalSampler

LOGGER = logging.getLogger(__name__)


class Dataset:
    def __init__(
            self,
            train_sampler,
            validation_sampler,
            test_sampler,
            num_items,
            max_sequence_length,
            item_frequencies
    ):
        self._train_sampler = train_sampler
        self._validation_sampler = validation_sampler
        self._test_sampler = test_sampler
        self._num_items = num_items
        self._max_sequence_length = max_sequence_length
        self._item_frequencies = item_frequencies

    @classmethod
    def create(cls, inter_json_path, max_sequence_length, sampler_type, is_extended=False):
        max_item_id = 0
        train_dataset, validation_dataset, test_dataset = [], [], []
        item_frequency_counts = None

        with open(inter_json_path, 'r') as f:
            user_interactions = json.load(f)

        for user_id_str, item_ids in user_interactions.items():
            user_id = int(user_id_str)

            if item_ids:
                max_item_id = max(max_item_id, max(item_ids))

            assert len(item_ids) >= 5, f'Core-5 dataset is used, user {user_id} has only {len(item_ids)} items'

            # Initialize frequency counter lazily once we know max id so far
            if item_frequency_counts is None:
                # We do not yet know final max, so start conservatively and grow if needed
                item_frequency_counts = {}

            # Count occurrences for frequency buckets
            for item_id in item_ids:
                item_frequency_counts[item_id] = item_frequency_counts.get(item_id, 0) + 1

            # sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (leave one out scheme, 8 - train, 9 - valid, 10 - test)
            if is_extended:
                # sample = [1, 2]
                # sample = [1, 2, 3]
                # sample = [1, 2, 3, 4]
                # sample = [1, 2, 3, 4, 5]
                # sample = [1, 2, 3, 4, 5, 6]
                # sample = [1, 2, 3, 4, 5, 6, 7]
                # sample = [1, 2, 3, 4, 5, 6, 7, 8]
                for prefix_length in range(2, len(item_ids) - 2 + 1):
                    train_dataset.append({
                        'user.ids': [user_id],
                        'item.ids': item_ids[:prefix_length],
                    })
            else:
                # sample = [1, 2, 3, 4, 5, 6, 7, 8]
                train_dataset.append({
                    'user.ids': [user_id],
                    'item.ids': item_ids[:-2],
                })

            # sample = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            validation_dataset.append({
                'user.ids': [user_id],
                'item.ids': item_ids[:-1],
            })

            # sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            test_dataset.append({
                'user.ids': [user_id],
                'item.ids': item_ids,
            })

        LOGGER.info(f'Train dataset size: {len(train_dataset)}')
        LOGGER.info(f'Validation dataset size: {len(validation_dataset)}')
        LOGGER.info(f'Test dataset size: {len(test_dataset)}')
        LOGGER.info(f'Max item id: {max_item_id}')

        train_sampler = TrainSampler(train_dataset, sampler_type, max_sequence_length=max_sequence_length)
        validation_sampler = EvalSampler(validation_dataset, max_sequence_length=max_sequence_length)
        test_sampler = EvalSampler(test_dataset, max_sequence_length=max_sequence_length)

        # Build dense frequency array of size num_items (ids are 0-indexed)
        num_items = max_item_id + 1
        item_frequencies = [0] * num_items
        if item_frequency_counts is not None:
            for item_id, cnt in item_frequency_counts.items():
                if 0 <= item_id < num_items:
                    item_frequencies[item_id] = cnt

        return cls(
            train_sampler=train_sampler,
            validation_sampler=validation_sampler,
            test_sampler=test_sampler,
            num_items=num_items,  # +1 added because our ids are 0-indexed
            max_sequence_length=max_sequence_length,
            item_frequencies=item_frequencies
        )

    def get_samplers(self):
        return self._train_sampler, self._validation_sampler, self._test_sampler

    @property
    def num_items(self):
        return self._num_items

    @property
    def max_sequence_length(self):
        return self._max_sequence_length

    @property
    def item_frequencies(self):
        return self._item_frequencies
