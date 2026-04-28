# План: воспроизведение Amazon-пайплайна для VK-LSVD

## Context

Цель — расширить дипломный эксперимент с одного датасета (Amazon Beauty) на VK-LSVD, чтобы получить второй независимый замер метрик качества для модификации семантических идентификаторов с CF-сигналом и logQ-коррекцией. Текущий пайплайн целиком построен под Amazon: эмбеды получаются из Llama-7b в [DatasetProcessing.ipynb](notebooks/DatasetProcessing.ipynb), позитивные пары — в [cf_dataset_builder.ipynb](tiger/cf_dataset_builder.ipynb), CF-finetune — в [cf_finetune.ipynb](tiger/cf_finetune.ipynb), квантизация — в [RQKmeansPipeline.ipynb](notebooks/RQKmeansPipeline.ipynb), и всё подаётся в TIGER через [tiger_kmeans_train_config.json](tiger/configs/tiger_kmeans_train_config.json).

VK-LSVD приятен тем, что у него уже есть готовые контентные эмбеды в `metadata/item_embeddings.npz`, а скачивание заведено в [LsvdDownload.ipynb](notebooks/LsvdDownload.ipynb), но он ни разу не запускался end-to-end. Нужно превратить полу-готовый download-ноутбук в полноценный аналог Amazon-пайплайна, не трогая код самого тайгера (чтобы не «убить весь смысл эксперимента»).

**Решения, согласованные с пользователем:**
- Сабсэмпл: `ur0.01_ir0.01` (1%×1%, ~77k юзеров / ~66k айтемов / ~1.4M интеракций; гарантированно влезает в A100).
- Раскладка: новая папка `../data/VK/` без префикса `vk_` (отступаем от формулировки в task.txt в пользу более чистой структуры; путь явно подтверждён через AskUserQuestion).
- Скоуп интеракций для `inter.json`: ВСЁ (base+gap+val+test), как в Amazon — TIGER сам делает leave-one-out по позиции.
- Код-лейаут: новые ноутбуки рядом со старыми, амазоновские не трогаем.

## Outputs (целевые файлы)

```
data/VK/
├── inter.json                       # {str(user_id): [item_id, ...]}, 0-index, dense
├── content_embeddings.pkl            # {'item_id': [...], 'embedding': [...]} — VK-эмбеды
├── positive_pairs.txt                # "anchor_id pos_id" по строке
├── tuned_content_embeddings.pkl      # тот же формат, после CF-finetune
└── tuned_index_rqkmeans.json         # {str(item_id): [c0,c1,c2(,c3)]}
```

Дополнительно: `tiger/configs/tiger_kmeans_train_config_vk.json` — копия амазон-конфига с путями на VK-файлы.

## Files to create

### 1. `notebooks/VkDatasetProcessing.ipynb` (аналог DatasetProcessing.ipynb + кусок LsvdDownload.ipynb)

Объединяет скачивание VK-LSVD с продакшн-форматом `inter.json` / `content_embeddings.pkl`, как в Amazon-пайплайне.

Шаги:
1. Скачать `metadata/*` и `subsamples/ur0.01_ir0.01/*` через `hf download` (берём cell-2 и cell-3 из [LsvdDownload.ipynb](notebooks/LsvdDownload.ipynb) как есть). Сохраняем в локальный `DATASET_PATH` — параметр в начале ноутбука; на A100 сервере это `/home/jovyan/vkml/data/vk_lsvd/raw`, на других машинах — переопределяется.
2. Прочитать ВСЕ 25 недельных parquet'ов (`week_00.parquet` … `week_24.parquet`), отфильтровать `timespent > 15` (POSITIVE_EVENT_TIMESPENT, как в LsvdDownload), отсортировать по `original_order`. Использовать функции `get_parquet_interactions` и логику конкатенации из [LsvdDownload.ipynb cell-5..cell-7](notebooks/LsvdDownload.ipynb).
3. **Core-5 фильтрация** (новое — в LsvdDownload её нет, но Amazon-пайплайн делает её обязательно): итеративно выбрасывать юзеров с <5 интеракций и айтемы с <5 интеракций до сходимости. Этот шаг есть в [DatasetProcessing.ipynb](notebooks/DatasetProcessing.ipynb), нужно перенести один в один.
4. Ремап `item_id` → 0-indexed dense (как в LsvdDownload cell-11), плюс ремап `user_id` → 0-indexed (для красоты ключей в `inter.json`; не критично, но сразу даёт компактность).
5. Группировка интеракций по юзеру с сортировкой по `original_order` (cell-14 из LsvdDownload), и сериализация в `inter.json` — `dict[str(user_id), list[int item_id]]`. Формат должен быть бит-в-бит совместим с тем, что читает [tiger/modeling/dataset/base.py](tiger/modeling/dataset/base.py) (string-ключи юзеров, list[int] значений; уже проверено для Amazon).
6. Загрузить `metadata/item_embeddings.npz` → отфильтровать по выжившим item'ам и переупорядочить по новому 0-indexed mapping. Сохранить в `content_embeddings.pkl` в **точно том же** формате, что и Amazon: `{'item_id': list[int], 'embedding': list[np.ndarray(D,) float32]}`. Размерность D возьмётся из VK как есть (для VK-LSVD это 32; cf_finetune и RQKmeans его подхватят автоматически).

### 2. `tiger/cf_dataset_builder_vk.ipynb` (аналог cf_dataset_builder.ipynb)

Перенести 1:1 из [cf_dataset_builder.ipynb](tiger/cf_dataset_builder.ipynb), заменив только пути:
- input: `../data/VK/inter.json`
- output: `../data/VK/positive_pairs.txt`

Логика `calc_pairs(history, drop_last_cnt=2)` со скользящим окном размера 2 — без изменений. После Core-5 у каждого юзера ≥5 элементов, поэтому drop_last=2 корректно даёт ≥3 пар (как в Amazon).

### 3. `tiger/cf_finetune_vk.ipynb` (аналог cf_finetune.ipynb)

Перенести 1:1 из [cf_finetune.ipynb](tiger/cf_finetune.ipynb), заменив только пути:
- inputs: `../data/VK/content_embeddings.pkl`, `../data/VK/positive_pairs.txt`
- output: `../data/VK/tuned_content_embeddings.pkl`

Архитектура (`TowerMLP`, `nt_xent_loss`, `tau=0.07`) и гиперпараметры остаются. Размерность `D` тянется из формы загруженных эмбедов (`num_items, D = X.shape`), так что 32-мерные VK-эмбеды отработают без правок. Если для VK потребуется отдельный sweep по τ или по числу эпох — это уже после first run'а.

⚠️ logQ-коррекции в `cf_finetune.ipynb` сейчас нет (стандартный NT-Xent без поправок). Это ок: согласно almanah.md, logQ — отдельная вариация, и для повторения ключевых результатов на VK достаточно tuned-варианта; logQ-эксперимент можно докинуть отдельно (out of scope этого плана).

### 4. `notebooks/RQKmeansPipelineVk.ipynb` (аналог RQKmeansPipeline.ipynb)

Перенести 1:1 из [RQKmeansPipeline.ipynb](notebooks/RQKmeansPipeline.ipynb), заменив только пути:
- input: `../data/VK/tuned_content_embeddings.pkl`
- output: `../data/VK/tuned_index_rqkmeans.json`

3 кодбука × 256 кластеров + collision_solver — без изменений. Формат выходного JSON `{str(item_id): list[int]}` остаётся бит-в-бит совместимым с тем, что читает TIGER (проверено по [Beauty/index_rqkmeans.json](data/Beauty/index_rqkmeans.json) — там 4 значения на айтем; collision-айтемы получают 4-й код, все остальные тоже добиваются до 4 кодов в текущей реализации).

### 5. `tiger/configs/tiger_kmeans_train_config_vk.json`

Копия [tiger_kmeans_train_config.json](tiger/configs/tiger_kmeans_train_config.json) с правками:
```json
{
  "experiment_name": "tiger_vk_kmeans",
  "dataset": {
    "inter_json_path":  "../data/VK/inter.json",
    "index_json_path":  "../data/VK/tuned_index_rqkmeans.json",
    "num_codebooks": 4,
    "max_sequence_length": 20,
    "sampler_type": "tiger"
  },
  ...
  "model": {
    ...
    "user_ids_count": 80000,   // было 2000; ~77k юзеров в VK сабсэмпле
    ...
  }
}
```

Все остальные гиперпараметры (`embedding_dim=128`, `codebook_size=256`, `num_codebooks=4`, и т.д.) сохраняем как у Amazon, чтобы конфигурация модели была сравнимой. `user_ids_count=80000` — модуль для хеша user-id в `BatchProcessor`; для Amazon (~22k юзеров) стояло 2000, для VK с ~77k нужно поднять, чтобы коллизий хеша было сопоставимо мало (это конфиг, а не код тайгера, — менять можно).

## Критические файлы (для справки при реализации)

- [notebooks/LsvdDownload.ipynb](notebooks/LsvdDownload.ipynb) — источник download/parquet-логики и week-разбиения.
- [notebooks/DatasetProcessing.ipynb](notebooks/DatasetProcessing.ipynb) — эталон Core-5 + ремапа + сериализации `inter.json` и `content_embeddings.pkl`.
- [tiger/cf_dataset_builder.ipynb](tiger/cf_dataset_builder.ipynb), [tiger/cf_finetune.ipynb](tiger/cf_finetune.ipynb), [notebooks/RQKmeansPipeline.ipynb](notebooks/RQKmeansPipeline.ipynb) — эталоны для VK-аналогов; меняются только пути.
- [tiger/modeling/dataset/base.py](tiger/modeling/dataset/base.py) — leave-one-out на позициях; диктует требование «всё-в-одной-последовательности».
- [tiger/modeling/dataloader/batch_processors.py](tiger/modeling/dataloader/batch_processors.py) — индексирует semantic codes по `item_id`; **требует, чтобы все айтемы из `inter.json` имели запись в `index_rqkmeans.json`** (важная инвариантa при Core-5 / ремапе).
- [data/Beauty/inter.json](data/Beauty/inter.json), [data/Beauty/index_rqkmeans.json](data/Beauty/index_rqkmeans.json) — образец схемы.

## Подводные камни

1. **Core-5 после timespent-фильтра.** LsvdDownload его не делает; без него короткие истории сломают `cf_dataset_builder` (требует ≥3 элементов после drop_last=2) и/или дадут юзеров без val/test. Делаем итеративно до сходимости, **обязательно после** ремапа на пересечение с эмбедами (иначе айтем-id 0..N-1 разъедется с эмбедами).
2. **Согласованность item-id во всех артефактах.** После Core-5+ремап один и тот же `item_id` должен индексировать одну и ту же строку в `content_embeddings.pkl`, один и тот же ключ в `tuned_index_rqkmeans.json` и один и тот же токен в `inter.json`. Делаем mapping один раз и применяем ко всем трём выходам.
3. **`max_id+1 == num_items`.** В `cf_finetune` `nn.Embedding(max_id+1, D)`; если айтемы плотные 0..N-1, всё ок — Core-5 + ремап это гарантирует.
4. **Размерность D≠4096.** Никаких хардкодов на 4096 в Amazon-пайплайне нет (проверено). VK-эмбеды (вероятно 32-мерные) пройдут без правок, но прогрев CF-finetune может занять иначе по числу шагов — это нормально, гиперы оставляем стандартные на первом проходе.
5. **`user_ids_count` в TIGER-конфиге.** Если оставить 2000 как у Amazon, при ~77k VK-юзеров будет много hash-коллизий. Поднимаем до ~80000.
6. **Локальные пути для скачивания.** `hf download` в LsvdDownload ходит через `HF_ENDPOINT="http://huggingface.proxy"` и пишет в `/home/jovyan/...`. Параметризуем `DATASET_PATH` в самом начале нового ноутбука.

## Verification (как проверять, что всё корректно)

End-to-end проверка по чек-листу, после каждого шага:

1. **После VkDatasetProcessing.ipynb**:
   - `inter.json`: `len(json.load(...))` ≈ количество выживших юзеров; `min(len(v) for v in d.values()) >= 5`; `max(max(v) for v in d.values()) == num_items - 1`.
   - `content_embeddings.pkl`: `len(d['item_id']) == len(d['embedding']) == num_items`; `d['item_id'] == list(range(num_items))`; все вектора одной размерности.
2. **После cf_dataset_builder_vk.ipynb**: открыть `positive_pairs.txt`, убедиться что обе колонки в диапазоне `[0, num_items-1]`, `anchor != positive` всюду, число строк ≈ ожидаемому (~суммарная длина историй − 2·num_users).
3. **После cf_finetune_vk.ipynb**: `tuned_content_embeddings.pkl` имеет ту же форму, что входной; loss убывает (выводится в ноутбуке); вектора L2-нормированы (`np.linalg.norm` ≈ 1).
4. **После RQKmeansPipelineVk.ipynb**: `tuned_index_rqkmeans.json` имеет ровно `num_items` ключей (`set(map(int, keys)) == set(range(num_items))`); все значения — листы длины 3 или 4 (после collision_solver) с элементами в `[0, 255]`.
5. **TIGER-прогон**: `python tiger/train_tiger.py --config tiger/configs/tiger_kmeans_train_config_vk.json` — запустить хотя бы 1 эпоху и убедиться, что:
   - dataloader не падает (значит inter.json и index валидны и согласованы),
   - train loss конечный и убывает,
   - val NDCG@5 / Recall@5 печатаются в стандартный лог.
   Сравнивать с Amazon-цифрами не требуем — это будет уже фаза эксперимента.

## Что делаем дальше (out of scope, но логичный next step)

После того как baseline-tuned для VK сойдётся, можно повторить logQ-вариант (как в almanah.md table) и добавить колонку «VK» в финальную таблицу диплома. Для этого нужно будет один раз доработать `cf_finetune.ipynb` под logQ-correction (добавить вычисление частот Q и поправку только в негативные слагаемые) и прогнать `cf_finetune_vk.ipynb` ещё раз. Этот шаг сейчас в план не входит.
