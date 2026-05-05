# План: cold / warm / hot метрики в `MastersDiploma/scripts/tiger/vk_train.py`

## Context

В `tiger/tiger_kmeans_baseline.ipynb` метрики по бакетам частоты айтема (cold/warm/hot) реализованы через параметр `allowed_item_mask=` в готовых классах `NDCGSemanticMetric` / `RecallSemanticMetric` из `tiger/modeling/metric/`:

```python
item_freqs_tensor = torch.tensor(dataset.item_frequencies, dtype=torch.long)
cold_mask = (item_freqs_tensor >= 5) & (item_freqs_tensor <= 10)
warm_mask = (item_freqs_tensor >= 5) & (item_freqs_tensor <= 50)
hot_mask  = (item_freqs_tensor >= 5) & (item_freqs_tensor <= 500)
...
'ndcg@5_cold': NDCGSemanticMetric(5, codebook_size, num_codebooks, allowed_item_mask=cold_mask),
```

**Но `MastersDiploma/scripts/tiger/vk_train.py` живёт в другом фреймворке (`irec`)**: датасет — это пред-сериализованные arrow-батчи `ArrowBatchDataset`, метрики считаются *внутри* [`TigerModel.forward()`](MastersDiploma/scripts/tiger/models.py) inline (строки 167–174), а наружу выдаются как тензоры формы `(batch_size,)` через колбек `BatchMetrics` + `MetricAccumulator`. Никакого `NDCGSemanticMetric` / `allowed_item_mask` тут нет — нужно перенести идею вручную.

## Целевой результат

В TensorBoard-логах `MastersDiploma/vk_tensorboard_logs/vk_baseline` появляются скаляры:

```
validation/recall@{5,10,20}_{cold,warm,hot}
validation/ndcg@{5,10,20}_{cold,warm,hot}
eval/recall@{5,10,20}_{cold,warm,hot}
eval/ndcg@{5,10,20}_{cold,warm,hot}
```

Без регрессий по уже имеющимся `recall@K` / `ndcg@K`. Бакетинг — по **target item** (тот самый next-item, который мы пытаемся предсказать), как это делает `allowed_item_mask` в Amazon-ноутбуке.

## Files to change

### 1. `MastersDiploma/scripts/tiger/vk_train.py`

#### 1.1. Посчитать частоты айтемов и собрать маски

Перед созданием `model = TigerModel(...)` (после загрузки `mappings`) добавить:

```python
import json
import numpy as np
import torch

INTER_JSON_PATH = os.path.join(IREC_PATH, 'data/VK/inter.json')

with open(INTER_JSON_PATH, 'r') as f:
    user_interactions = json.load(f)

# Частоты считаем по ОБУЧАЮЩЕЙ части истории (item_ids[:-2]) —
# это совпадает с тем, как dataset.item_frequencies формируется в Amazon-пайплайне
# (val=item_ids[-2], test=item_ids[-1] не учитываем, чтобы не «подсматривать» в таргет).
num_items = 0
freqs = {}
for items in user_interactions.values():
    for it in items:
        num_items = max(num_items, it + 1)
    for it in items[:-2]:
        freqs[it] = freqs.get(it, 0) + 1

freq_arr = np.zeros(num_items, dtype=np.int64)
for it, c in freqs.items():
    freq_arr[it] = c

item_freqs = torch.from_numpy(freq_arr)
cold_mask = ((item_freqs >= 5) & (item_freqs <= 10))
warm_mask = ((item_freqs >= 5) & (item_freqs <= 20))
hot_mask  = ((item_freqs >= 5) & (item_freqs <= 100))

logger.debug(f'num_items={num_items}, cold={int(cold_mask.sum())}, '
             f'warm={int(warm_mask.sum())}, hot={int(hot_mask.sum())}')

item_buckets = {'cold': cold_mask, 'warm': warm_mask, 'hot': hot_mask}
```

#### 1.2. Прокинуть маски в модель

В вызов `TigerModel(...)` добавить новый kwarg:

```python
model = TigerModel(
    ...
    logits_processor=partial(...),
    item_buckets=item_buckets,   # NEW
).to(DEVICE)
```

#### 1.3. Расширить колбеки

В обоих `cb.Validation(...)` (validation и eval) — для `BatchMetrics.metrics` лямбды и `MetricAccumulator.accumulators`:

```python
metric_names = []
for k in (5, 10, 20):
    metric_names += [f'recall@{k}', f'ndcg@{k}']
    for bucket in ('cold', 'warm', 'hot'):
        metric_names += [f'recall@{k}_{bucket}', f'ndcg@{k}_{bucket}']

# в BatchMetrics:
metrics=lambda model_outputs, _: {
    'loss': model_outputs['loss'].item(),
    **{name: model_outputs[name].tolist() for name in metric_names},
},
# в MetricAccumulator:
accumulators={
    'validation/loss': cb.MeanAccumulator(),
    **{f'validation/{name}': cb.MeanAccumulator() for name in metric_names},
},
```

(Для eval-валидатора — то же самое с префиксом `eval/`.)

`EarlyStopping(metric='eval/ndcg@20', ...)` оставляем как есть — основная метрика не меняется.

### 2. `MastersDiploma/scripts/tiger/models.py`

#### 2.1. Принять буфер масок

В `TigerModel.__init__`:

```python
def __init__(self, ..., logits_processor=None, item_buckets=None):
    ...
    self.logits_processor = logits_processor
    self._bucket_names = []
    if item_buckets is not None:
        self._bucket_names = list(item_buckets.keys())
        for name, mask in item_buckets.items():
            # register_buffer чтобы маска ехала на DEVICE вместе с .to(DEVICE)
            self.register_buffer(f'_bucket_mask_{name}', mask.bool(), persistent=False)
```

Параметр **опциональный** с дефолтом `None` — амазонский [`train.py`](MastersDiploma/scripts/tiger/train.py) и [`varka.py`](MastersDiploma/scripts/tiger/varka.py) работают без изменений.

#### 2.2. Считать бакетные метрики в forward

В блоке `if not self.training:` после строки `metrics[f'ndcg@{k}'] = ...`:

```python
# Raw (pre-semantic) target item id для бакетинга.
# labels.ids кладётся в arrow-батч через vk_varka.py:save_batches_to_arrow
# (он сохраняет ВСЕ ключи batch, в т.ч. исходные labels.ids формы (B, 1)).
target_item_ids = inputs['labels.ids'][:, 0]  # (B,)

for bucket in self._bucket_names:
    bucket_mask = getattr(self, f'_bucket_mask_{bucket}')  # (num_items,)
    in_bucket = bucket_mask[target_item_ids]               # (B,)
    for k in (5, 10, 20):
        # hits/recall уже посчитаны выше в том же цикле — их надо переиспользовать.
        # Самый чистый вариант — сразу вынести вычисление в внешний цикл и
        # параллельно строить и базовые, и бакетные метрики (см. ниже).
        ...
```

Чтобы не пересчитывать `hits` дважды, переписать существующий цикл `for k in [5, 10, 20]:` так:

```python
target_item_ids = inputs['labels.ids'][:, 0] if self._bucket_names else None
in_bucket = {
    name: getattr(self, f'_bucket_mask_{name}')[target_item_ids]
    for name in self._bucket_names
} if self._bucket_names else {}

for k in [5, 10, 20]:
    hits = (all_hits[:, :k] == self._sem_id_len).float()        # (B, k)
    recall = hits.sum(dim=-1)                                   # (B,)
    discount_factor = 1 / torch.log2(
        torch.arange(1, k + 1, 1).float() + 1.
    ).to(hits.device)                                           # (k,)
    ndcg = torch.einsum('bk,k->b', hits, discount_factor)        # (B,)

    metrics[f'recall@{k}'] = recall.cpu().float()
    metrics[f'ndcg@{k}']   = ndcg.cpu().float()

    for name, mask in in_bucket.items():
        # Возвращаем only-in-bucket сэмплы; пустой батч → пустой тензор,
        # MeanAccumulator его пропустит (см. подводный камень №2).
        metrics[f'recall@{k}_{name}'] = recall[mask].cpu().float()
        metrics[f'ndcg@{k}_{name}']   = ndcg[mask].cpu().float()
```

### 3. `MastersDiploma/scripts/tiger/vk_varka.py` — проверить, не менять

`save_batches_to_arrow` итерирует `batch.items()` и сохраняет всё подряд — `labels.ids` (raw 0-indexed item id формы `(B, 1)`, кладётся в [`data.py:EvalDataset.__getitem__`](MastersDiploma/scripts/tiger/data.py)) уезжает в arrow без правок. **Но это нужно явно проверить** на одном существующем валид-батче (см. Verification §1) — если нет, добавить в keep-list, иначе forward упадёт по KeyError.

## Critical files (для справки)

- [`MastersDiploma/scripts/tiger/models.py`](MastersDiploma/scripts/tiger/models.py) — там `TigerModel.forward` со встроенным расчётом метрик (строки 130–181).
- [`MastersDiploma/scripts/tiger/data.py`](MastersDiploma/scripts/tiger/data.py) — `EvalDataset` (строки 151–178): источник `labels.ids` как raw item_id.
- [`MastersDiploma/scripts/tiger/vk_varka.py`](MastersDiploma/scripts/tiger/vk_varka.py) — пайплайн пред-сериализации arrow-батчей (`save_batches_to_arrow` на строке 144).
- [`MastersDiploma/scripts/tiger/vk_train.py`](MastersDiploma/scripts/tiger/vk_train.py) — точка изменений по callbacks (строки 119–200).
- [`tiger/tiger_kmeans_baseline.ipynb`](tiger/tiger_kmeans_baseline.ipynb) — эталон порогов (cells `d14f549c`, `826980a0`).

## Подводные камни

1. **`labels.ids` обязан быть в arrow-батче.** Если по какой-то причине `vk_varka.py` его не сохраняет (например, из-за фильтрации внутри `Collate`), `inputs['labels.ids']` упадёт. Проверить распаковав один `.arrow` через `feather.read_table` (см. Verification §1) **до** правки `models.py`. Если ключа нет — поправить `vk_varka.py`, добавив явное сохранение.
2. **Пустой бакет в батче.** `recall[mask].cpu().float()` для `mask.sum() == 0` даёт пустой тензор → `.tolist()` → `[]`. `cb.MeanAccumulator()` из irec должен обрабатывать пустой list (просто не апдейтить mean). **Проверить** на dummy-прогоне: если падает с делением на 0 — обернуть `[]` в `[float('nan')]` либо завести аналог `MeanAccumulator`, игнорирующий NaN/empty. Если irec уже это умеет — менять ничего не нужно. Это единственная неизвестная в плане; решается на месте за 5 минут.
3. **Маски на правильном device.** `register_buffer` + `.to(DEVICE)` решает; индексирование `mask[target_item_ids]` требует одного device у обоих тензоров. `target_item_ids` приходит из batch'a, который уже на `DEVICE` (см. `ToDevice(DEVICE)` в `vk_train.py:65`).
4. **Item id 0-indexed dense.** Гарантируется `VkDatasetProcessing.ipynb` (Core-5 + ремап). Если когда-то нарушится — `freq_arr[it] = c` упадёт по out-of-bounds. Надёжная защита уже есть в `data.py:Dataset.create` через `assert len(item_ids) >= 5`.
5. **Частоты считаем по `[:-2]`, не по всей истории.** Это соответствует `dataset.item_frequencies` в Amazon-ноутбуке, где fitness-сигнал — только train. Если посчитать по всей истории, в `cold` могут попасть айтемы, которые видны только в test'е, и метрика начнёт дрейфовать туда, куда не нужно.
6. **Совместимость с амазонским `train.py`.** Параметр `item_buckets=None` дефолтный, в forward — `if self._bucket_names:` no-op. Никаких изменений в `train.py` / `varka.py` не требуется.
7. **`hot ⊃ warm ⊃ cold` по порогам.** Это по дизайну Amazon-ноутбука: `hot` включает в себя cold-и-warm айтемы, бакеты не дизъюнктны. Не «исправляем» — иначе сломается прямая сопоставимость с `tiger_kmeans_baseline.ipynb`.

## Verification

Проверять последовательно, не запуская длительный train заранее:

1. **Sanity-check arrow-батчей** (до правок кода). Прочитать один файл из `data/VK/tiger_valid_batches/`:

   ```python
   import pyarrow.feather as feather
   t = feather.read_table('data/VK/tiger_valid_batches/batch_000000_len_<X>.arrow')
   print(t.column_names)  # должно содержать 'labels.ids'
   ```

   Если `labels.ids` нет — поправить `vk_varka.py` и пересобрать батчи (или добавить keep-list). Иначе можно идти дальше.

2. **Sanity-check масок** (после §1.1). Логирование при старте `vk_train.py`:

   ```
   num_items=~66000, cold=~30000, warm=~50000, hot=~63000
   ```

   Точные числа зависят от VK ur0.01_ir0.01 сабсэмпла; проверять, что `cold ≤ warm ≤ hot ≤ num_items` и все три > 0.

3. **Smoke-train на 1 эпоху** (`NUM_EPOCHS = 1`, `EPOCH_NUM_STEPS = 64`). После прогона:
   - В TensorBoard появились все 18 новых скаляров (3 k × 2 metric × 3 bucket).
   - Базовые `validation/recall@K`, `validation/ndcg@K` равны тому, что было до правок (детерминизм при `fix_random_seed(42)`).

4. **Качественная санити-проверка после ~10 эпох**: `recall@K_hot ≥ recall@K_warm ≥ recall@K_cold` (популярные айтемы предсказываются легче). Это эмпирическое наблюдение из Amazon-ноутбука, на VK должно повториться; иначе — повод подозревать ошибку в индексировании (например, `item_freqs_tensor` посчитан по полной истории, а не по `[:-2]`).

5. **Полный train-run.** Запустить `vk_train.py` до сходимости, забрать `best_checkpoint`. Сверить итоговые числа с теми, что были без бакетных метрик: ничего не должно сместиться (новые тензоры считаются только в eval-режиме и не влияют на градиент).

## Out of scope

- Никаких правок в `tiger/` (амазонский) и в `tiger/modeling/...`. Меняем только MastersDiploma-репозиторий.
- Не вводим четвёртый бакет «super-hot» (>500) и не делаем дизъюнктное разбиение — иначе теряется сопоставимость с эталонным ноутбуком.
- logQ-вариация для VK — отдельный план (см. финальный раздел `vk_plan.md`); сюда не примешиваем.
