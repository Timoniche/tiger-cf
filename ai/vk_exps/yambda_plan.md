# План: поддержка Yambda-экспериментов в tiger-cf + MastersDiploma

## Context

Цель — расширить дипломный замер с двух датасетов (Amazon Beauty, VK-LSVD) на третий — Yandex Yambda ([yandex/yambda](https://huggingface.co/datasets/yandex/yambda)). Yambda приятен тем, что у него уже есть нормализованные эмбеды (`embeddings.parquet`, `normalized_embed`, dim=64 — как у VK). Скачивание заведено в [ai/vk_exps/YambdaDownload.ipynb](YambdaDownload.ipynb), но он ни разу не превращался в TIGER-совместимые `inter.json` / `content_embeddings.pkl`.

**Инвариант (из task.txt):** код тайгера менять нельзя.

**Решения, заморожены по согласованию с пользователем:**
- Источник: `flat/50m/likes`, intersect с `embeddings.parquet`.
- Сжатие сразу в processing-ноутбуке: `USER_SUBSAMPLE_RATIO=0.05` + `MAX_HISTORY_PER_USER=15`. Таргет ~50k юзеров / ~20-30k айтемов / ~500k интеракций — как `data/VK_small/`.
- Раскладка: `data/yambda/` (без префикса/суффикса `small` — Yambda изначально единственный сабсэмпл).
- Имена файлов 1:1 как в VK_small.

## Outputs

```
data/yambda/
├── raw/                              # выход hf_hub_download
│   ├── embeddings.parquet
│   └── flat/50m/likes.parquet
├── inter.json                        # {str(user_id): [item_id, ...]}, dense 0-index
├── content_embeddings.pkl            # {'item_id': [...], 'embedding': [...]}, D=64
├── positive_pairs.txt
├── item_frequencies.txt
├── tuned_content_embeddings.pkl
├── logq_tuned_content_embeddings.pkl
├── tuned_index_rqkmeans.json         # для оригинального tiger-репо
└── new_format_index_rqkmeans.json    # для MastersDiploma
```

## Созданные файлы

1. [notebooks/YambdaDatasetProcessing.ipynb](../../notebooks/YambdaDatasetProcessing.ipynb) — скачивание из HF, intersect с эмбедами, random-сабсэмпл юзеров (`USER_SUBSAMPLE_RATIO`), truncation последних `K` интеракций, итеративный Core-5, dense remap, сериализация `inter.json` + `content_embeddings.pkl`.
2. [tiger/cf_dataset_builder_yambda.ipynb](../../tiger/cf_dataset_builder_yambda.ipynb) — `positive_pairs.txt` + `item_frequencies.txt` (`base_dir = '../data/yambda/'`).
3. [tiger/cf_finetune_yambda.ipynb](../../tiger/cf_finetune_yambda.ipynb) — TowerMLP + NT-Xent (`tau=0.07`, 4 epochs), `D` тянется из формы pkl.
4. [tiger/cf_finetune_log_q_yambda.ipynb](../../tiger/cf_finetune_log_q_yambda.ipynb) — то же + `nt_xent_loss_with_logq_neg_only`.
5. [MastersDiploma/YambdaRQKmeansPipeline.ipynb](../../MastersDiploma/YambdaRQKmeansPipeline.ipynb) — RQ-KMeans (3 codebook × 256 + collision_solver) → `new_format_index_rqkmeans.json`.
6. [MastersDiploma/scripts/tiger/yambda_varka.py](../../MastersDiploma/scripts/tiger/yambda_varka.py) — копия `vk_varka.py` с `data_prefix='data/yambda/'`.
7. [MastersDiploma/scripts/tiger/yambda_train.py](../../MastersDiploma/scripts/tiger/yambda_train.py) — копия `vk_train.py` с `EXPERIMENT_NAME='yambda_baseline'`, `logs_folder='yambda_tensorboard_logs'`. Cold/warm/hot бакеты пересчитываются автоматически по `inter.json`.
8. [tiger/configs/tiger_kmeans_train_config_yambda.json](../../tiger/configs/tiger_kmeans_train_config_yambda.json) — пути на `../data/yambda/`, `user_ids_count=80000`.

## Порядок прогонки

1. `notebooks/YambdaDatasetProcessing.ipynb` → `data/yambda/inter.json` + `content_embeddings.pkl`.
2. `tiger/cf_dataset_builder_yambda.ipynb` → `positive_pairs.txt` + `item_frequencies.txt`.
3. `tiger/cf_finetune_yambda.ipynb` → `tuned_content_embeddings.pkl` (baseline).
4. `tiger/cf_finetune_log_q_yambda.ipynb` → `logq_tuned_content_embeddings.pkl` (logQ).
5. `MastersDiploma/YambdaRQKmeansPipeline.ipynb` → `new_format_index_rqkmeans.json` (вход = `tuned_content_embeddings.pkl`).
6. `python MastersDiploma/scripts/tiger/yambda_varka.py` → arrow-батчи.
7. `python MastersDiploma/scripts/tiger/yambda_train.py` → train в `yambda_tensorboard_logs/yambda_baseline`.

logQ-вариант через шаги 5-7 с заменой входного pkl и отдельным `EXPERIMENT_NAME`.

## Подводные камни

1. **Объём `flat/50m/likes`.** Без сабсэмпла Core-5 + remap съедают много RAM. Дефолт `R=0.05` и `K=15` рассчитан под VK_small-объём.
2. **`embeddings.parquet` ~1.5GB.** Скачивание `hf_hub_download` за один запрос; параметризован `DATASET_PATH`.
3. **`normalized_embed` уже L2-нормирован.** В `cf_finetune_yambda` всё равно идёт `F.normalize` (идемпотентный).
4. **`max_id+1 == num_items`.** Гарантирует Core-5 + dense remap в processing-ноутбуке.
5. **`labels.ids` обязан быть в arrow-батче.** Проверено для VK; для Yambda `yambda_varka.py` использует тот же `save_batches_to_arrow`.
6. **`user_ids_count` в TIGER-конфиге.** При `~50k` юзеров стоит `80000`. Если возьмём больше — поднять.

## Out of scope

- Sweep по `tau`, `K`, `R`. Фиксируем VK-эталон.
- TIGER-прогон с logQ-индексом — отдельный `tiger_kmeans_train_config_yambda_logq.json` + второй запуск `yambda_train.py` (под отдельным `EXPERIMENT_NAME='yambda_logq'`).
- `multi_event` / `listens` интеракции — пока только `likes`.
- Полный `flat/50m` без сабсэмпла — это десятки часов на A100.
