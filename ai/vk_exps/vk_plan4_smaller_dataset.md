# План: ускорение VK-пайплайна до 1-2 часов

## Context

Текущее состояние пайплайна VK (см. [vk_plan.md](vk_plan.md), [vk_plan_2.md](vk_plan_2.md), [vk_plan_3.md](vk_plan_3.md)):
- субсэмпл `ur0.01_ir0.01` уже скачан и обработан в `data/VK/inter.json` (14 MB);
- `cf_dataset_builder_vk.ipynb`, `cf_finetune_vk.ipynb`, `cf_finetune_log_q_vk.ipynb`, `RQKmeansPipelineVk.ipynb` готовы и завязаны на `data/VK/...`;
- TIGER-обучение идёт через [`MastersDiploma/scripts/tiger/vk_varka.py`](MastersDiploma/scripts/tiger/vk_varka.py) (предсериализация arrow-батчей) → [`vk_train.py`](MastersDiploma/scripts/tiger/vk_train.py).

**Проблема пользователя:** varka 25 минут, TIGER 6 часов, метрика `ndcg@20` ещё растёт. Хочется 1-2 часа.

**Где время на самом деле горит.** В [`vk_varka.py:204`](MastersDiploma/scripts/tiger/vk_varka.py) и [`tiger/modeling/dataset/base.py:53`](tiger/modeling/dataset/base.py) стоит `is_extended=True` — для каждого юзера генерится по одному train-сэмплу на каждый префикс длины `2..len-2`. Числа на текущем `data/VK/inter.json`:

| параметр                                | значение     |
| --------------------------------------- | ------------ |
| users                                   | 49 691       |
| total interactions                      | 1 284 340    |
| mean / median / p90 / p99 / max history | 26 / 16 / 58 / 129 / 250 |
| **train samples (`is_extended=True`)**  | **1 135 267** |

Сравнение с Amazon Beauty (~22k users × ~9 items avg → ~130k train samples) показывает, что VK-пайплайн прокачивает ~9× больше градиентных шагов на эпоху. Отсюда и 6 часов.

Меньший субсэмпл с HF (`ur0.001`) пользователь явно отверг как «слишком мало». Значит, режем уже скачанный `ur0.01_ir0.01` локально, не трогая код TIGER (это инвариант из [task.txt:52](task.txt)).

## Стратегия

Два независимых рычага, оба применяются на уровне `inter.json` — никаких правок в TIGER-коде/конфигах не требуется:

1. **Cap per-user history до последних `K` интеракций** (`MAX_HISTORY_PER_USER`).
   - Внутри `is_extended=True` число train-сэмплов на юзера = `min(len, K) - 3`. Сжатие хвоста режет квадратично-выглядящий рост (юзеры с 250 событиями давали 247 сэмплов).
   - Качество практически не страдает: `max_sequence_length=20` в [`vk_train.py:27`](MastersDiploma/scripts/tiger/vk_train.py) и так обрезает до 20 элементов на forward — ранние интеракции в обучении уже не видны, мы лишь не плодим лишние префиксы.

2. **Случайный сабсэмпл юзеров до доли `R`** (`USER_SUBSAMPLE_RATIO`).
   - Линейное сокращение train-сэмплов и числа cf-пар. Распределение длин историй сохраняется (берём случайных, не топ-N).
   - Сразу даёт меньшую `varka.py` (она доминируется числом юзеров).

### Оценка для разных `K` и `R` (ground truth, посчитал на текущем `data/VK/inter.json`):

| `K`  | `R`   | users   | train_samples | % от исходного | прогноз TIGER |
| ---- | ----- | ------- | ------------- | -------------- | ------------- |
| —    | 1.0   | 49 691  | 1 135 267     | 100%           | **~6 ч (текущий)** |
| 30   | 1.0   | 49 691  | 745 258       | 66%            | ~4 ч          |
| 20   | 1.0   | 49 691  | 573 005       | 50%            | ~3 ч          |
| 15   | 1.0   | 49 691  | 454 475       | 40%            | ~2.4 ч        |
| 12   | 1.0   | 49 691  | 368 182       | 32%            | ~2 ч          |
| 20   | 0.5   | 24 845  | 350 579       | 31%            | **~1.9 ч ✓**  |
| 20   | 0.4   | 19 876  | 284 608       | 25%            | **~1.5 ч ✓**  |
| 15   | 0.5   | 24 845  | ~227 000      | 20%            | **~1.2 ч ✓**  |
| 12   | 0.5   | 24 845  | ~184 000      | 16%            | ~1 ч          |

**Дефолт по согласованию с пользователем: `K=15, R=1.0`** (все ~50k юзеров, ≈454k train-сэмплов, прогноз ~2.4 ч).

- `K=15` чуть жёстче, чем `MAX_SEQ_LEN=20`: модель на forward видит максимум 14 элементов (последний идёт в target), что для VK с медианой 16 — ровно граница, за которой начинается длинный хвост (p90=58, p99=129). Cap отрезает именно его.
- Юзеров не трогаем — сохраняем диверсити для cold/warm/hot бакетов и для оценки `varka` без сдвига популяционной статистики.
- `K` и `R` остаются параметрами в верхушке нового ноутбука, чтобы можно было свинговать, если 2.4ч окажется много (тогда добавить `R=0.5` → ~1.2ч).

### Что НЕ делаем (и почему)

- **Не трогаем `is_extended` в `vk_varka.py`.** Переключение на `False` даст 1 сэмпл/юзер (~50k всего, ~25× ускорение), но обедняет градиентный сигнал и ломает сопоставимость с Amazon-конфигом. Это уже не «тот же эксперимент на меньших данных», а другой режим обучения.
- **Не уменьшаем модель** (`embedding_dim`, `num_heads`, etc.) — это опять ломает сопоставимость с Amazon в дипломе.
- **Не пилим `NUM_EPOCHS=300`** в [`vk_train.py:26`](MastersDiploma/scripts/tiger/vk_train.py) явно — там стоит `EarlyStopping(metric='eval/ndcg@20', patience=40, ...)`, которая сама остановит тренировку, как только метрика перестанет расти на меньшем датасете. Достаточно сократить данные.
- **Не используем «топ-N юзеров по длине истории»** — внесёт смещение в сторону power-users и сломает оценку на cold/warm.

## Целевые файлы

```
data/VK_small/
├── inter.json                          # NEW: усечённый и/или сабсэмпленный
├── content_embeddings.pkl               # NEW: отфильтрованный по выжившим item_id
├── positive_pairs.txt                   # NEW: cf_dataset_builder_vk выход
├── item_frequencies.txt                 # NEW: cf_dataset_builder_vk выход (для logQ)
├── tuned_content_embeddings.pkl         # NEW: cf_finetune_vk выход
├── logq_tuned_content_embeddings.pkl    # NEW (опц.): cf_finetune_log_q_vk выход
├── tuned_index_rqkmeans.json            # NEW: RQKmeansPipelineVk выход
└── new_format_index_rqkmeans.json       # NEW (опц.): для MastersDiploma vk_train.py
```

Имя `VK_small` (а не `VK_tiny`/`VK_subsample`) — короткое и читаемое; `VK/` остаётся для возможности сравнения старого и нового результатов.

## Files to create / change

### 1. `notebooks/VkDatasetShrink.ipynb` — НОВЫЙ ноутбук

Один компактный ноутбук, который применяется поверх готовых артефактов в `data/VK/`. Не пересобирает HF-датасет, не пересчитывает эмбеды — только режет.

Шаги (одна-две ячейки на каждый):

#### 1.1. Параметры в самом верху

```python
import os, json, pickle, random
import numpy as np

INPUT_DIR  = '../data/VK'
OUTPUT_DIR = '../data/VK_small'

MAX_HISTORY_PER_USER = 15      # K
USER_SUBSAMPLE_RATIO = 1.0     # R; 1.0 = не сабсэмплируем (текущий дефолт)
RANDOM_SEED = 42
KEEP_LAST = True               # True = items[-K:], False = items[:K]; всегда True для рекомендералки
```

#### 1.2. Загрузить inter.json и проверить инварианты

```python
with open(os.path.join(INPUT_DIR, 'inter.json'), 'r') as f:
    user_interactions = json.load(f)

orig_users = len(user_interactions)
orig_items = max(max(v) for v in user_interactions.values()) + 1
print(f'before: users={orig_users}, num_items={orig_items}, total_inter={sum(len(v) for v in user_interactions.values())}')
```

#### 1.3. Сабсэмпл юзеров и truncation

```python
random.seed(RANDOM_SEED)
user_ids = list(user_interactions.keys())
if USER_SUBSAMPLE_RATIO < 1.0:
    keep_n = int(len(user_ids) * USER_SUBSAMPLE_RATIO)
    user_ids = random.sample(user_ids, keep_n)

shrunk = {}
for uid in user_ids:
    seq = user_interactions[uid]
    seq = seq[-MAX_HISTORY_PER_USER:] if KEEP_LAST else seq[:MAX_HISTORY_PER_USER]
    shrunk[uid] = seq
```

#### 1.4. Re-Core-5 + dense remap (id-ы юзеров и айтемов)

После truncation некоторые айтемы могут потерять интеракции, и появятся юзеры с <5 элементов (если изначально были на грани). Делаем итеративный Core-5 до сходимости — точно как в [`notebooks/DatasetProcessing.ipynb`](notebooks/DatasetProcessing.ipynb), плюс ремап в плотный 0..N-1:

```python
def core5_filter(d):
    while True:
        item_counts = {}
        for seq in d.values():
            for it in seq:
                item_counts[it] = item_counts.get(it, 0) + 1
        bad_items = {it for it, c in item_counts.items() if c < 5}
        new_d = {}
        for u, seq in d.items():
            cleaned = [it for it in seq if it not in bad_items]
            if len(cleaned) >= 5:
                new_d[u] = cleaned
        if new_d == d:
            return new_d
        d = new_d

shrunk = core5_filter(shrunk)

# Dense remap item_ids → 0..M-1, user_ids → 0..N-1
old_items = sorted({it for seq in shrunk.values() for it in seq})
item_remap = {old: new for new, old in enumerate(old_items)}
old_users = sorted(shrunk.keys(), key=lambda x: int(x))
user_remap = {old: new for new, old in enumerate(old_users)}

remapped = {
    str(user_remap[u]): [item_remap[it] for it in seq]
    for u, seq in shrunk.items()
}

new_users = len(remapped)
new_items = len(old_items)
new_inter = sum(len(v) for v in remapped.values())
print(f'after: users={new_users}, num_items={new_items}, total_inter={new_inter}')
```

#### 1.5. Сохранить inter.json и отфильтрованные эмбеды

```python
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, 'inter.json'), 'w') as f:
    json.dump(remapped, f)

# content_embeddings.pkl: {'item_id': [...], 'embedding': [np.ndarray, ...]}
with open(os.path.join(INPUT_DIR, 'content_embeddings.pkl'), 'rb') as f:
    emb = pickle.load(f)

old_to_new = item_remap
emb_by_old = dict(zip(emb['item_id'], emb['embedding']))

new_emb = {'item_id': list(range(new_items)), 'embedding': [None] * new_items}
for old_id, new_id in old_to_new.items():
    new_emb['embedding'][new_id] = emb_by_old[old_id]
assert all(e is not None for e in new_emb['embedding'])

with open(os.path.join(OUTPUT_DIR, 'content_embeddings.pkl'), 'wb') as f:
    pickle.dump(new_emb, f)
```

#### 1.6. Печать прогноза train-сэмплов (sanity)

```python
extended_samples = sum(max(0, len(v) - 3) for v in remapped.values())
print(f'TIGER train samples (is_extended=True): {extended_samples:,}')
print(f'reduction vs original: {extended_samples / 1_135_267 * 100:.1f}%')
# для дефолтных K=15, R=1.0 ожидаем ~454k (~40% от исходного)
```

Этот единственный ноутбук — точка управления всем сжатием. После его запуска всё ниже по конвейеру просто переключается на `data/VK_small/`.

### 2. Параметризация существующих VK-ноутбуков

Идея — **не делать копий** для `VK_small`, а ввести ровно одну переменную пути в каждом из существующих VK-ноутбуков, чтобы переключаться `'../data/VK'` ⇄ `'../data/VK_small'` за одну правку. Так артефакты копий ноутбуков не плодятся, а пользователь явно решает, какую версию пайплайна гонять.

Файлы, требующие минимальной правки (одна переменная-путь в первой ячейке):

- [`tiger/cf_dataset_builder_vk.ipynb`](tiger/cf_dataset_builder_vk.ipynb) — заменить `inter_json_path = '../data/VK/inter.json'` на `DATA_DIR = '../data/VK_small'` + дальше через `os.path.join`. Выход: `data/VK_small/positive_pairs.txt` + `data/VK_small/item_frequencies.txt`.
- [`tiger/cf_finetune_vk.ipynb`](tiger/cf_finetune_vk.ipynb) — то же. Вход: `content_embeddings.pkl`, `positive_pairs.txt`. Выход: `tuned_content_embeddings.pkl`.
- [`tiger/cf_finetune_log_q_vk.ipynb`](tiger/cf_finetune_log_q_vk.ipynb) — то же. Доп.вход: `item_frequencies.txt`. Выход: `logq_tuned_content_embeddings.pkl`.
- [`notebooks/RQKmeansPipelineVk.ipynb`](notebooks/RQKmeansPipelineVk.ipynb) — то же. Вход: `tuned_content_embeddings.pkl` (или logQ-вариант). Выход: `tuned_index_rqkmeans.json`.

Если в каких-то из этих ноутбуков сейчас пути захардкожены прямо в литералах (`'../data/VK/...'`), стоит вытащить общую `DATA_DIR` в первую ячейку (по аналогии с `VkDatasetShrink.ipynb`). Это ровно по одной правке на ноутбук.

### 3. `MastersDiploma/scripts/tiger/vk_varka.py` и `vk_train.py` — параметризация путей

В обоих скриптах сейчас:

```python
data_prefix = 'data/VK/'
index_path  = 'data/VK/new_format_index_rqkmeans.json'
```

Меняем на одно место наверху файла:

```python
DATA_PREFIX = os.environ.get('VK_DATA_PREFIX', 'data/VK_small/')
INDEX_PATH  = os.environ.get('VK_INDEX_PATH', f'{DATA_PREFIX}new_format_index_rqkmeans.json')
```

Это позволяет:
- по умолчанию (без env) гонять усечённый VK_small;
- одной env-переменной откатиться на полный `data/VK/`.

`EXPERIMENT_NAME = 'vk_baseline'` лучше тоже подменить через env (`VK_EXPERIMENT_NAME`), чтобы tensorboard-логи усечённого и полного прогона не схлопнулись в один эксперимент. Аналогично для `checkpoints/`.

**Конфиг тайгера для оригинального репозитория ([`tiger/configs/tiger_kmeans_train_config_vk.json`](tiger/configs/tiger_kmeans_train_config_vk.json))**: создать рядом `tiger_kmeans_train_config_vk_small.json` с теми же гиперами, только `inter_json_path` и `index_json_path` → `../data/VK_small/...`. `user_ids_count` поднять до `30000` (было 80000 для полного VK; на сабсэмпле ~25k юзеров достаточно). Остальное копируем 1:1 — задача не правка модели, а ускорение.

## Critical files

- [`MastersDiploma/scripts/tiger/vk_varka.py`](MastersDiploma/scripts/tiger/vk_varka.py) — рендерит arrow-батчи; именно тут `is_extended=True` плодит N×prefix сэмплов.
- [`MastersDiploma/scripts/tiger/vk_train.py`](MastersDiploma/scripts/tiger/vk_train.py) — `EarlyStopping(metric='eval/ndcg@20', patience=40)`. Не нужно вручную крутить `NUM_EPOCHS=300`, EarlyStopping сделает работу.
- [`tiger/modeling/dataset/base.py:53-65`](tiger/modeling/dataset/base.py) — источник `is_extended` логики.
- [`tiger/modeling/dataset/samplers.py:27-38`](tiger/modeling/dataset/samplers.py) — `_last_item_transform`, режет до `max_sequence_length` элементов на forward.
- [`notebooks/DatasetProcessing.ipynb`](notebooks/DatasetProcessing.ipynb) — эталон Core-5 + dense remap. Логика 1:1 копируется в `VkDatasetShrink.ipynb`.
- [`notebooks/VkDatasetProcessing.ipynb`](notebooks/VkDatasetProcessing.ipynb) — там сейчас лежит исходный VK-ремап; полезно пересмотреть, чтобы remap-семантика в `VkDatasetShrink.ipynb` совпала бит-в-бит.

## Подводные камни

1. **Re-Core-5 после truncation обязателен.** Если юзер из `data/VK/` имел 5 интеракций все из «головы», после truncation `KEEP_LAST=True` его выкинет. Аналогично, айтемы, которые жили только в «хвостах» удалённых юзеров, потеряют все интеракции. Без re-Core-5 ассерт `assert len(item_ids) >= 5` в `Dataset.create:41` упадёт.

2. **Item-id ремап обязан быть согласован с эмбедами.** После Core-5+remap новый `item_id == 0` должен индексировать тот же эмбед, что старый `item_remap_inverse[0]` в `content_embeddings.pkl`. В §1.5 это обеспечено явным `new_emb['embedding'][new_id] = emb_by_old[old_id]`. Если этого не сделать, contrastive learning будет считать loss на расфазированных эмбедах — ndcg на TIGER упадёт молча.

3. **`USER_SUBSAMPLE_RATIO` фиксирует случайность через `random.seed(42)`.** Менять seed между прогонами `cf_finetune_vk` и `vk_varka` нельзя — они должны видеть один и тот же `inter.json`. Поэтому seed строго в одном месте — `VkDatasetShrink.ipynb`.

4. **`user_ids_count` в TIGER-конфиге = размер хеш-таблицы для user-id.** Для `K=15, R=1.0` число юзеров остаётся ~50k (Core-5 после truncation отрежет малую долю), поэтому оставляем `user_ids_count=80000` в `tiger_kmeans_train_config_vk_small.json` без изменений. На MastersDiploma-стороне `NUM_USER_HASH = 2000` в [`vk_train.py:33`](MastersDiploma/scripts/tiger/vk_train.py) — оно безразмерно к числу юзеров (хешируется через murmur, коллизии нормальны). Не трогаем без отдельного тикета.

5. **`varka` (irec, не путать с `vk_varka.py`) тоже ускорится автоматически.** Пользователь упомянул её в task4 («varka обучается 25 минут»). Если речь о MastersDiploma `varka.py` — она ходит в тот же `inter.json`, поэтому переключение на `data/VK_small/` сократит её слабо (юзеров столько же при `R=1.0`, экономия только на коротких историях, ~5-10%). Если хочется заметного ускорения varka — подключить `R<1.0` отдельным шагом.

6. **logQ-вариант после сжатия.** Частоты должны пересчитаться на новом `inter.json` — поэтому после `VkDatasetShrink.ipynb` сначала прогоняется `cf_dataset_builder_vk.ipynb` (создаёт новый `item_frequencies.txt`), и только потом `cf_finetune_log_q_vk.ipynb`. Старый `data/VK/item_frequencies.txt` не годится — id-ы там другие после remap.

7. **Тайгеровский конфиг полагается на dense `[0, num_items)`.** Любой gap в id-ах после shrink сломает индексирование `mapping_tensor[ids]` в [`SemanticIdsMapper`](MastersDiploma/scripts/tiger/vk_varka.py:111). Dense remap в §1.4 это гарантирует — но если когда-то решим не remap'ить, нужно явно проверять `set(ids) == set(range(N))` перед прогоном.

8. **TensorBoard: разные `EXPERIMENT_NAME`.** `vk_baseline` в `vk_train.py` сейчас захардкожен. Если запустить на VK_small без переименования, новые скаляры лягут поверх старого прогона на одном и том же имени эксперимента — кривые в TB смешаются и сравнения не получится. Нужно поменять либо вручную (на `vk_baseline_small`), либо через env.

## Verification

Проверять последовательно — не запускать длительный TIGER, пока не сошлись numbers ниже.

1. **После `VkDatasetShrink.ipynb`**:
   - В выводе печатается `before / after`. Должно быть `users_after ≤ users_before * USER_SUBSAMPLE_RATIO + малая дельта от Core-5` и `total_inter_after ≤ total_inter_before`.
   - `extended_samples` совпадает с предсказанием из таблицы выше (для дефолтных `K=15, R=1.0` ожидаем ~454k, ~40% от исходного).
   - `data/VK_small/inter.json` существует, `min(len(v) for v in d.values()) >= 5`, `set(int(k) for k in d.keys()) == set(range(num_users))`, `set(it for v in d.values() for it in v) == set(range(num_items))`.
   - `data/VK_small/content_embeddings.pkl` имеет `len(d['item_id']) == num_items`, все эмбеды размерности D из исходного pkl.

2. **После `cf_dataset_builder_vk.ipynb` на VK_small**:
   - `data/VK_small/positive_pairs.txt`: число строк ≈ `Σ max(0, len-2)` ≈ для дефолтных `K=15, R=1.0` ~530k.
   - `data/VK_small/item_frequencies.txt`: `wc -l == num_items`. Сумма частот ≈ `Σ max(0, len-2)`.

3. **После `cf_finetune_vk.ipynb` на VK_small**:
   - Loss убывает ровно так же, как на полном VK (та же кривая по форме, просто меньше шагов на эпоху).
   - Выходной pkl читается, L2-нормы ≈ 1.

4. **После `RQKmeansPipelineVk.ipynb` на VK_small**:
   - `tuned_index_rqkmeans.json` содержит ровно `num_items` ключей; все коды в `[0, 255]`.

5. **TIGER smoke-test (1 эпоха)**:
   - Экспортировать `VK_DATA_PREFIX='data/VK_small/'`, `VK_EXPERIMENT_NAME='vk_baseline_small'`.
   - `python MastersDiploma/scripts/tiger/vk_varka.py` — проверить, что arrow-батчи в `data/VK_small/tiger_train_batches/` создались. Число файлов ≈ `extended_samples / TRAIN_BATCH_SIZE` ≈ 454k/256 ≈ 1775 (для дефолтных `K=15, R=1.0`).
   - `python MastersDiploma/scripts/tiger/vk_train.py` на 1 эпоху (через `cb.StopAfterNumSteps(EPOCH_NUM_STEPS)` или `NUM_EPOCHS=1`): один проход по тренировке + одна валидация — должно укладываться в ~2-3 минуты на A100.

6. **Полный TIGER-прогон**:
   - Засечь wall-clock. Цель: меньше 2.5 часов до остановки по EarlyStopping. Если упирается в 6 часов так же, как раньше — значит EarlyStopping не сработал (метрика ещё растёт), и надо дальше уменьшать данные (`K=15, R=0.5` → ~1.2ч; `K=12, R=0.5` → ~1ч).
   - Сверить итоговые `eval/ndcg@20`, `eval/recall@20` (и cold/warm/hot бакеты) с тем, что было на полном VK. Падение на 5-15% — ожидаемо (меньше данных, меньше юзеров). Сильное падение (>30%) — повод подозревать ошибку в remap'е или Core-5.

## Out of scope

- **Запуск всех downstream-нотбуков прямо в этом плане.** План добавляет один новый ноутбук + параметризацию путей в существующих. Запуск/прогон до конечных метрик — следующий шаг.
- **Sweep по `K` и `R`.** Фиксируем дефолт `K=15, R=1.0`. Если 2.4ч не хватит — крутим в одном месте (`K=15, R=0.5` или `K=12, R=1.0`) и пересобираем.
- **Логика `is_extended`.** Не меняем (см. «Что НЕ делаем»).
- **Cold/warm/hot пороги.** Они в [`vk_train.py:79-81`](MastersDiploma/scripts/tiger/vk_train.py) считаются по `inter.json` динамически на старте — на VK_small пересчитаются автоматически, ничего не правим.
- **Замена `ur0.01_ir0.01` на меньший HF-субсэмпл.** Пользователь явно отверг (`ur0.001` слишком мало), плюс это требует пересчёта эмбедов и full re-download.
- **`logq_tuned_index_rqkmeans.json` + отдельный TIGER-конфиг для logQ-варианта на VK_small.** Аналогично `vk_plan_3.md` — это отдельная пара (`RQKmeansPipelineVk` на logQ-pkl + `tiger_kmeans_train_config_vk_small_logq.json`). В этот план не входит.
