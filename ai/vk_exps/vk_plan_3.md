# План: VK CF-finetune с logQ-коррекцией

## Context

Для Amazon-датасета logQ-вариант контрастного обучения уже реализован в [tiger/cf_finetune.ipynb](tiger/cf_finetune.ipynb): функция `nt_xent_loss_with_logq_neg_only(z1, z2, a_ids, p_ids, logq, tau=0.07)` вычитает `logq` только из *негативных* колонок логит-матрицы (диагональ-позитив возвращается обратно), что даёт несмещённую оценку softmax-знаменателя по подвыборке негативов и при этом не штрафует true-positive за популярность.

Для VK уже есть базовый CF-finetune в [tiger/cf_finetune_vk.ipynb](tiger/cf_finetune_vk.ipynb) (NT-Xent без коррекции). Цель — рядом с ним положить **отдельный** ноутбук [tiger/cf_finetune_log_q_vk.ipynb](tiger/cf_finetune_log_q_vk.ipynb), который повторяет логику Amazon-варианта `nt_xent_loss_with_logq_neg_only`, но на VK-данных.

**Блокер:** logQ-коррекция требует файл `item_frequencies.txt` с частотами айтемов в обучающей части истории (`history[:-2]`). Для Amazon он генерируется в [tiger/cf_dataset_builder.ipynb](tiger/cf_dataset_builder.ipynb) (cells `691adbe6` и `fa60e287`), но в VK-версии [tiger/cf_dataset_builder_vk.ipynb](tiger/cf_dataset_builder_vk.ipynb) этот шаг отсутствует. **Сначала** нужно добавить генерацию частот в VK-builder, **потом** писать logQ-ноутбук.

## Целевые файлы

```
data/VK/
├── item_frequencies.txt                # NEW: одна строка = count по каждому item_id, 0..num_items-1
└── logq_tuned_content_embeddings.pkl   # NEW: выход logQ-варианта тюнинга

tiger/
├── cf_dataset_builder_vk.ipynb         # MODIFY: добавить генерацию item_frequencies.txt
└── cf_finetune_log_q_vk.ipynb          # NEW
```

Имя выходного pkl выбрано по аналогии с Amazon-ноутбуком (`logq_tuned_content_embeddings_temp_0_07.pkl` там сохраняется в финальной cell `043f5e11`); для VK сохраним короткий вариант `logq_tuned_content_embeddings.pkl`, чтобы было симметрично уже существующему `tuned_content_embeddings.pkl`. Параметризовать temperature в имени пока не нужно — фиксируем `tau=0.07` как и в эталоне.

## Files to change / create

### 1. `tiger/cf_dataset_builder_vk.ipynb` — добавить генерацию `item_frequencies.txt`

В существующий ноутбук, **между** ячейкой загрузки `inter.json` (cell `8306abd4`, та где считается `max_item_id`) и ячейкой с `def calc_pairs(...)` (cell `5d7bc0c5`), добавить две новые ячейки 1:1 как в Amazon-builder:

```python
# cell 1 — путь
item_frequencies_path = '../data/VK/item_frequencies.txt'
```

```python
# cell 2 — подсчёт частот по history[:-2] (без val/test)
item_frequency_counts = {}
for user_id_str, history in user_interactions.items():
    usable_items = history[:-2] if len(history) > 2 else []
    for item_id in usable_items:
        item_frequency_counts[item_id] = item_frequency_counts.get(item_id, 0) + 1

num_items = max_item_id + 1
item_frequencies = [0] * num_items
for item_id, count in item_frequency_counts.items():
    if item_id < 0 or item_id >= num_items:
        raise ValueError(f'Invalid item ID: {item_id}')
    item_frequencies[item_id] = count

with open(item_frequencies_path, 'w', encoding='utf-8') as f:
    for count in item_frequencies:
        f.write(f'{count}\n')

print(f'item_frequencies.txt сохранён: {item_frequencies_path} ({num_items} строк)')
```

После этого нужно **перезапустить** [cf_dataset_builder_vk.ipynb](tiger/cf_dataset_builder_vk.ipynb) end-to-end, чтобы `data/VK/item_frequencies.txt` физически появился.

### 2. `tiger/cf_finetune_log_q_vk.ipynb` — новый ноутбук

Копия [tiger/cf_finetune_vk.ipynb](tiger/cf_finetune_vk.ipynb) с тремя точечными правками:

#### 2.1. Заголовок и пути

```python
# первая markdown-ячейка
# VK CF finetune с logQ-коррекцией

# Аналог `tiger/cf_finetune_vk.ipynb`, но с logQ-коррекцией только по негативам
# (см. `tiger/cf_finetune.ipynb`, функция `nt_xent_loss_with_logq_neg_only`).
# Идея — в логит-матрице вычитаем log Q(j) из всех негативных колонок,
# а диагональ (true positive) оставляем как есть. Это даёт несмещённый
# softmax-знаменатель по подвыборке негативов и не штрафует true-positive за популярность.
```

```python
# ячейка с путями — добавляем item_frequencies_path и меняем имя выходного файла
embeddings_input_path = '../data/VK/content_embeddings.pkl'
pairs_path = '../data/VK/positive_pairs.txt'
item_frequencies_path = '../data/VK/item_frequencies.txt'
tuned_embeddings_output_path = '../data/VK/logq_tuned_content_embeddings.pkl'
```

#### 2.2. Загрузка частот и подсчёт `logq`

После cell с `device = ...` (тот же блок, где `tower = tower.to(device)`; в Amazon-ноутбуке это cell `7b0658d4`), добавить ячейку **точно как в Amazon** (`659c071c` + `ceec94b2`):

```python
counts_list = []
with open(item_frequencies_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        counts_list.append(int(line))
counts = torch.tensor(counts_list, dtype=torch.float32, device=device)

if counts.numel() != max_id + 1:
    raise ValueError('Some item_ids popularities missed')

total_count = counts.sum()
q = counts / torch.clamp(total_count, min=1.0)
logq = torch.log(torch.clamp(q, min=1e-12))
print(f'num_items={counts.numel()}, total_count={int(total_count.item())}')
```

`max_id` уже определён выше (cell `c11b00e7` в `cf_finetune_vk.ipynb`).

#### 2.3. Функция лосса и тренировочный цикл

Заменить `nt_xent_loss(...)` на `nt_xent_loss_with_logq_neg_only(...)` ровно в той форме, как в Amazon-ноутбуке (cell `d019d343`):

```python
def nt_xent_loss_with_logq_neg_only(z1, z2, a_ids, p_ids, logq, tau=0.07):
    B = z1.size(0)
    device = z1.device
    idx = torch.arange(B, device=device)

    base12 = (z1 @ z2.T) / tau
    col_logq = logq[p_ids].to(device)
    logits12 = base12 - col_logq.unsqueeze(0)
    logits12[idx, idx] = base12[idx, idx]

    base21 = (z2 @ z1.T) / tau
    row_logq = logq[a_ids].to(device)
    logits21 = base21 - row_logq.unsqueeze(0)
    logits21[idx, idx] = base21[idx, idx]

    labels = idx
    return 0.5 * (F.cross_entropy(logits12, labels) +
                  F.cross_entropy(logits21, labels))
```

И в обучающем цикле:

```python
loss = nt_xent_loss_with_logq_neg_only(zA, zP, a_ids, p_ids, logq, tau=0.07)
```

Всё остальное (TowerMLP-архитектура, AdamW(lr=3e-4, wd=1e-4), `B=32`, `num_epochs=4`, gradient clip 1.0, финальная инференс-петля) — **без изменений**.

#### 2.4. Сохранение тюненых эмбедов

Та же логика, что и в `cf_finetune_vk.ipynb` (cell `512d23dd` + `d8fabca2`), только output путь другой — `tuned_embeddings_output_path = '../data/VK/logq_tuned_content_embeddings.pkl'` (см. §2.1).

## Critical files (для справки)

- [tiger/cf_finetune.ipynb](tiger/cf_finetune.ipynb) — эталон logQ-логики (cells `659c071c`, `ceec94b2`, `d019d343`, `767959d5`).
- [tiger/cf_finetune_vk.ipynb](tiger/cf_finetune_vk.ipynb) — базовый VK-ноутбук, от которого делаем копию.
- [tiger/cf_dataset_builder.ipynb](tiger/cf_dataset_builder.ipynb) — эталон генерации `item_frequencies.txt` (cells `691adbe6`, `fa60e287`).
- [tiger/cf_dataset_builder_vk.ipynb](tiger/cf_dataset_builder_vk.ipynb) — текущий VK-builder, в котором не хватает генерации частот.

## Подводные камни

1. **Частоты считаем по `history[:-2]`, не по полной истории.** Это критично: `[:-2]` исключает val/test и даёт ту же популярность, что использует Amazon-вариант. Если посчитать по всей истории, в logQ попадёт «подсмотренная» статистика.
2. **`counts.numel() == max_id + 1`.** Один в один из Amazon-ноутбука: если айтемы плотные `0..N-1` (а это гарантирует Core-5 в [VkDatasetProcessing.ipynb](notebooks/VkDatasetProcessing.ipynb)), всё ок. Иначе loss упадёт по out-of-bounds в `logq[p_ids]`.
3. **Айтемы с `count == 0`.** Возможны: айтем участвует в эмбедах и в `inter.json`, но во всех его юзерских историях он стоит в `[-2:]`. Тогда `q == 0`, `log(clamp(q, 1e-12))` даёт `log(1e-12) ≈ -27.6`, и в lossе он получит **бонус** (вычитаем большой отрицательный = прибавляем большой положительный). Это в точности то же поведение, что и в Amazon-ноутбуке (там тот же `clamp(q, min=1e-12)`); специально не «исправляем», чтобы воспроизвести оригинал бит-в-бит. Если в логах `cf_finetune_log_q_vk.ipynb` после тренировки увидим, что лосс взрывается — придётся вернуться и исследовать сколько таких айтемов; пока считаем что Amazon-эталон уже это пережил, и для VK поведёт себя так же.
4. **`tau=0.07` фиксированный.** В Amazon-ноутбуке закомментированы варианты с `tau=1.0`; в финальной версии остался `0.07`. Для VK берём ровно `0.07` ради сопоставимости с Amazon-результатами; sweep по τ — отдельная задача.
5. **`logq` на правильном device.** В функции `nt_xent_loss_with_logq_neg_only` делается `logq[p_ids].to(device)`, так что даже если `logq` создан на CPU, `.to(device)` его перенесёт. Но создаём сразу на `device` (как в Amazon cell `659c071c`) — экономит на per-batch копировании.
6. **`base_emb.weight[item_ids] = F.normalize(...)`.** Унаследовано из `cf_finetune_vk.ipynb` (cell `b021be53`). Работает корректно при плотных 0-indexed item_ids; не трогаем.
7. **Sanity-check на дубликаты пар (`a == b`).** Уже есть в `cf_finetune_vk.ipynb` (cell `0667539a`), сохраняем. logQ-коррекция диагональ возвращает обратно, поэтому self-pairs (если бы они были) попали бы и в позитив, и в шум — отдельно их фильтровать не нужно, но если cnt > 0 — это сигнал, что `cf_dataset_builder_vk.ipynb` сломан.
8. **Совместимость с downstream-пайплайном.** Output `logq_tuned_content_embeddings.pkl` имеет ту же схему (`{'item_id': [...], 'embedding': [...]}`), что и `tuned_content_embeddings.pkl` — значит [notebooks/RQKmeansPipelineVk.ipynb](notebooks/RQKmeansPipelineVk.ipynb) подхватит его без правок, нужно только перенаправить input-путь, когда будем гонять logQ-вариант через RQ-Kmeans.

## Verification

1. **После правки `cf_dataset_builder_vk.ipynb`**:
   - Запустить ноутбук end-to-end.
   - Проверить, что `data/VK/item_frequencies.txt` существует и `wc -l data/VK/item_frequencies.txt == num_items` (где `num_items` = `max_item_id + 1`, выводится в самом ноутбуке).
   - Сумма частот должна быть `≈ Σ max(0, len(history) − 2)` по всем юзерам — не критично проверять точно, но порядок величины должен сходиться (для VK ur0.01_ir0.01 это где-то ~1.3M).
   - Распределение по бакетам имеет смысл (для проверки): `(freq >= 5) & (freq <= 10)` — десятки тысяч айтемов; полностью «холодных» (`freq < 5`) — большинство для VK-LSVD из-за длинного хвоста.

2. **После создания `cf_finetune_log_q_vk.ipynb`**:
   - Sanity-check загрузки: `counts.numel() == max_id + 1` не падает.
   - `logq` имеет форму `(num_items,)`, без `nan`/`inf` (clamp на `1e-12` это гарантирует).
   - Запуск 1 эпохи: loss убывает. На VK с `tau=0.07` ожидаемые цифры порядка тех же, что у Amazon — ~12 → ~11 за 4 эпохи; точные числа другие, но монотонное убывание обязательно.
   - На выходе `data/VK/logq_tuned_content_embeddings.pkl` имеет ту же форму, что input: `len(d['item_id']) == len(d['embedding']) == num_items`, размерность каждого вектора `D`.
   - Спот-чек L2-нормы тюненых векторов: `np.linalg.norm(d['embedding'][0]) ≈ 1.0` (`TowerMLP.forward` нормирует).

3. **Сравнение с базовым VK-вариантом** (опционально, не блокер): после прогонки `logq_tuned_content_embeddings.pkl` через `RQKmeansPipelineVk.ipynb` и далее через TIGER-train (`tiger_kmeans_train_config_vk.json`, поменять `index_json_path` на logQ-выход) — итоговые `recall@K` / `ndcg@K` на VK должны быть сопоставимы или лучше, чем у `tuned_content_embeddings.pkl`. Это уже не часть текущего плана, но логичный next-step.

## Out of scope

- **TIGER-прогон с logQ-индексом.** Конфиг `tiger_kmeans_train_config_vk.json` сейчас указывает на `tuned_index_rqkmeans.json`. Чтобы оценить logQ-вариант на VK, нужна отдельная пара: `RQKmeansPipelineVk.ipynb` → `logq_tuned_index_rqkmeans.json` → `tiger_kmeans_train_config_vk_logq.json`. Это **не входит** в этот план — сюда входит только подготовка `logq_tuned_content_embeddings.pkl`.
- **Sweep по `tau` / числу эпох / размеру батча.** Сейчас фиксируем эталонные значения Amazon-ноутбука.
- **Изменения в `cf_finetune.ipynb` / `cf_finetune_vk.ipynb`.** Эти ноутбуки не трогаем — logQ-вариант для VK строго в новом файле `cf_finetune_log_q_vk.ipynb`.
- **Альтернативные формы logQ** (full logQ, не только negatives). В Amazon-ноутбуке закомментированы варианты `nt_xent_loss_with_logq` (full) и обычный `nt_xent_loss`. Для VK воспроизводим **только** `_neg_only` — единственный «живой» вариант на момент cell `767959d5`.
