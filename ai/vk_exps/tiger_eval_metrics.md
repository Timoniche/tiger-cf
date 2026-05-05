# Как TIGER считает Recall@K и NDCG@K на train/eval

Отчёт по коду из [MastersDiploma/scripts/tiger/](../../MastersDiploma/scripts/tiger/) и [MastersDiploma/src/irec/](../../MastersDiploma/src/irec/). Цель — пояснить, что именно сравнивается с чем, кто играет роль ground truth, и почему «семантический идентификатор не сразу весь собирается».

---

## 0. Картинка целиком (TL;DR)

1. У каждого пользователя есть последовательность item_id: `[i1, i2, ..., iN]`. Через **leave-one-out** последний айтем — таргет для test/eval, предпоследний — для validation, всё до них — train history.
2. Каждый item_id заранее (через RQ-Kmeans) превращён в **семантический идентификатор** — кортеж из 4 кодов (`num_codebooks=4`, каждый код 0..255). Маппинг лежит в `index_rqkmeans.json`.
3. Энкодер T5 принимает входную последовательность семантических кодов истории + хеш юзера. Декодер должен авторегрессивно сгенерировать 4 кода таргет-айтема.
4. **На train** считается обычная cross-entropy по 4 позициям next-token. Метрики не считаются.
5. **На eval/validation** запускается **beam search** (30 лучей → 20 возвращаемых последовательностей). Каждая «гипотеза» — это 4 кода (один полный семантический ID). Логитс-процессор гарантирует, что эти 4 кода соответствуют реально существующему айтему **и** не одному из уже посещённых пользователем айтемов.
6. Recall@K = доля примеров, где среди top-K гипотез нашёлся **точный** семантический ID таргета (равенство всех 4 кодов). NDCG@K — стандартный DCG (1/log2(rank+1)) на той же бинарной матрице попаданий, разделённый на ideal-DCG для одного релевантного айтема (= 1, упрощается).

То есть «семантический ID собирается покодово» только на стороне декодера: beam search строит его по одному коду за шаг, а сравниваем уже целиком — predicted (B, K, 4) против label (B, 1, 4).

---

## 1. Откуда берётся ground truth: leave-one-out в [data.py](../../MastersDiploma/scripts/tiger/data.py)

Файл [scripts/tiger/data.py:31-95](../../MastersDiploma/scripts/tiger/data.py#L31-L95) — `Dataset.create`. Для каждого юзера из `inter.json`:

```python
# data.py:62-78
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
```

Пометка комментариев в коде ровно отражает leave-one-out:
- train видит `[i1..i_{N-2}]`,
- validation видит `[i1..i_{N-1}]` (предсказывает `i_{N-1}`),
- eval видит `[i1..i_N]` (предсказывает `i_N`).

Дальше `EvalDataset.__getitem__` на [data.py:163-178](../../MastersDiploma/scripts/tiger/data.py#L163-L178) формирует «вход» и «таргет»:

```python
# data.py:163-178
def __getitem__(self, index):
    sample = self._dataset[index]

    item_sequence = sample['item.ids'][-self._max_sequence_length:][:-1]
    next_item = sample['item.ids'][-self._max_sequence_length:][-1]

    return {
        'user.ids': np.array(sample['user.ids'], dtype=np.int64),
        'user.length': np.array([len(sample['user.ids'])], dtype=np.int64),
        'item.ids': np.array(item_sequence, dtype=np.int64),
        'item.length': np.array([len(item_sequence)], dtype=np.int64),
        'labels.ids': np.array([next_item], dtype=np.int64),
        'labels.length': np.array([1], dtype=np.int64),
        'visited.ids': np.array(sample['item.ids'][:-1], dtype=np.int64),
        'visited.length': np.array([len(sample['item.ids'][:-1])], dtype=np.int64),
    }
```

То есть для validation:
- `item.ids` = `[i1..i_{N-2}]` — это **энкодерный вход**
- `labels.ids` = `[i_{N-1}]` — **ground truth айтем** (длина 1!)
- `visited.ids` = `[i1..i_{N-2}]` — список «уже виденных» айтемов, чтобы их **исключить** из beam search

Аналогично для eval, только `next_item = i_N`, `visited = [i1..i_{N-1}]`.

Train-семпл строится через `_last_item_transform` ([data.py:132-142](../../MastersDiploma/scripts/tiger/data.py#L132-L142)) — структурно то же самое (предсказать последний айтем подсеквенса), но без `visited.ids` и без расчёта метрик в forward.

> **Важно:** ground truth — это всегда **один айтем** (`labels.length = 1`). Никакого «правильно угадать топ-K» не существует на стороне таргета — таргет уникален, K варьирует только число гипотез модели.

---

## 2. Из item_id в семантический ID: [varka.py](../../MastersDiploma/scripts/tiger/varka.py)

`varka.py` — это **офлайн-препроцессор**, который превращает «логические» батчи (`item.ids`, `labels.ids`) в готовые `.arrow`-батчи с семантическими кодами и кладёт их в `tiger_train_batches/`, `tiger_valid_batches/`, `tiger_eval_batches/`. Дальше `train.py` уже только читает arrow и подаёт в модель.

### 2.1. SemanticIdsMapper — собственно маппинг

[varka.py:117-139](../../MastersDiploma/scripts/tiger/varka.py#L117-L139):

```python
class SemanticIdsMapper(Transform):
    def __init__(self, mapping, names=[]):
        super().__init__()
        self._mapping = mapping
        self._names = names

        data = []
        for i in range(len(mapping)):
            data.append(mapping[str(i)])
        self._mapping_tensor = torch.tensor(data, dtype=torch.long)
        self._semantic_length = self._mapping_tensor.shape[-1]

    def __call__(self, batch):
        for name in self._names:
            if f'{name}.ids' in batch:
                ids = batch[f'{name}.ids']
                lengths = batch[f'{name}.length']
                assert ids.min() >= 0
                assert ids.max() < self._mapping_tensor.shape[0]
                batch[f'{name}.semantic.ids'] = self._mapping_tensor[ids].flatten().numpy()
                batch[f'{name}.semantic.length'] = lengths * self._semantic_length

        return batch
```

`mapping` — это уже знакомый `index_rqkmeans.json` вида `{"0": [c0,c1,c2,c3], "1": [...], ...}`. Этот трансформ применяется и к `item` (история), и к `labels` (таргет). После него:

- `batch['item.semantic.ids']` — flatten-массив длины `len(history) * 4`
- `batch['labels.semantic.ids']` — flatten-массив длины `1 * 4 = 4` (ровно 4 кода таргета)
- `batch['labels.semantic.length'] = [4]`

### 2.2. ToMasked — паддинг до общей длины батча

[varka.py:80-114](../../MastersDiploma/scripts/tiger/varka.py#L80-L114) — стандартный right-aligned паддинг. После него:

- `batch['item.semantic.padded']` — `(B, max_seq_len * 4)`
- `batch['labels.semantic.padded']` — `(B, 4)` (одинаково для всех, т.к. таргет — 1 айтем)

### 2.3. TigerProcessing — финальная склейка для T5

[varka.py:46-77](../../MastersDiploma/scripts/tiger/varka.py#L46-L77):

```python
class TigerProcessing(Transform):
    def __call__(self, batch):
        input_semantic_ids, attention_mask = batch['item.semantic.padded'], batch['item.semantic.mask']
        batch_size = attention_mask.shape[0]

        input_semantic_ids[~attention_mask] = PAD_TOKEN_ID

        input_semantic_ids = np.concat([
            input_semantic_ids,
            NUM_CODEBOOKS * CODEBOOK_SIZE + batch['user.hashed.ids'][:, None]
        ], axis=-1)

        attention_mask = np.concat([
            attention_mask,
            np.ones((batch_size, 1), dtype=attention_mask.dtype)
        ], axis=-1)

        batch['input.data'] = input_semantic_ids
        batch['input.mask'] = attention_mask

        target_semantic_ids = batch['labels.semantic.padded']
        target_semantic_ids = np.concat([
            np.ones(
                (batch_size, 1),
                dtype=np.int64,
            ) * DECODER_START_TOKEN_ID,
            target_semantic_ids
        ], axis=-1)

        batch['output.data'] = target_semantic_ids

        return batch
```

После этого:
- `input.data` — `(B, max_seq_len * 4 + 1)` — последовательность семантических кодов истории + 1 токен с хешем юзера
- `output.data` — `(B, 5)` — `[DECODER_START, sem0, sem1, sem2, sem3]` для таргета

> Заметка про вокабуляр: семантические коды для разных кодбуков **сдвинуты** в едином вокабуляре (`unified_vocab_size = codebook_size * sem_id_len + user_ids_count + 10`, см. [models.py:108](../../MastersDiploma/scripts/tiger/models.py#L108)). Код 0 кодбука 0 ≠ код 0 кодбука 1 — у них разные глобальные ID. Это потом активно используется в `CorrectItemsLogitsProcessor` для запрета «не своих» позиций.

---

## 3. Train: cross-entropy по 4-ём позициям семантического ID

[models.py:137-153](../../MastersDiploma/scripts/tiger/models.py#L137-L153):

```python
def forward(self, inputs):
    input_semantic_ids = inputs['input.data']
    attention_mask = inputs['input.mask']
    target_semantic_ids = inputs['output.data']

    decoder_input_ids = target_semantic_ids[:, :-1].contiguous()
    labels = target_semantic_ids[:, 1:].contiguous()

    model_output = self.model(
        input_ids=input_semantic_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        labels=labels
    )
    loss = model_output['loss']

    metrics = {'loss': loss.detach()}
```

Тут чисто стандартная T5-схема:
- `decoder_input_ids = [DECODER_START, sem0, sem1, sem2]` — `(B, 4)`
- `labels = [sem0, sem1, sem2, sem3]` — `(B, 4)`

T5 учится в 4 шагах next-token: при входе `[DEC_START]` предсказать `sem0`, при `[DEC_START, sem0]` — `sem1`, и т.д. **На train recall/NDCG не считаются вообще** — только loss. Метрики из строки `metrics = {'loss': ...}` потом подхватываются callback'ом `cb.BatchMetrics` в [train.py:166-168](../../MastersDiploma/scripts/tiger/train.py#L166-L168).

---

## 4. Eval: beam search и сборка предсказания покодово

[models.py:155-200](../../MastersDiploma/scripts/tiger/models.py#L155-L200):

```python
if not self.training:
    visited_batch = inputs['visited.padded']

    output = self.model.generate(
        input_ids=input_semantic_ids,
        attention_mask=attention_mask,
        num_beams=self._num_beams,                  # 30
        num_return_sequences=self._num_return_sequences,  # 20
        max_length=self._sem_id_len + 1,            # 5 = DEC_START + 4 кода
        decoder_start_token_id=self.config.decoder_start_token_id,
        eos_token_id=self.config.eos_token_id,
        pad_token_id=self.config.pad_token_id,
        do_sample=False,
        early_stopping=False,
        logits_processor=[self.logits_processor(visited_items=visited_batch)] if self.logits_processor is not None else [],
    )

    predictions = output[:, 1:].reshape(-1, self._num_return_sequences, self._sem_id_len)

    all_hits = (torch.eq(predictions, labels[:, None]).sum(dim=-1))  # (batch_size, top_k)
```

Здесь и проявляется «покодовая сборка»:
- `model.generate` запускает beam search на 4 шага (max_length=5, минус decoder_start = 4 шага декодирования).
- На каждом шаге `t∈{0,1,2,3}` декодер выдаёт распределение над всем `unified_vocab_size`. Логитс-процессор (см. п. 5) обнуляет всё, кроме валидных кодов кодбука `t` и только тех префиксов, которые продолжают **существующий** семантический ID и не принадлежат уже посещённому айтему.
- Beam search ведёт 30 параллельных лучей; на выходе сохраняются 20 лучших по log-вероятности.

После генерации:
- `output` имеет форму `(B * num_return_sequences, 5)` (с decoder_start в начале)
- `output[:, 1:]` → `(B * 20, 4)` — собранные семантические ID кандидатов
- `predictions = ... .reshape(-1, 20, 4)` → `(B, 20, 4)`

`labels` тут — это `target_semantic_ids[:, 1:]` (см. п. 3) формы `(B, 4)`. Через `labels[:, None]` получаем `(B, 1, 4)`.

`all_hits = (predictions == labels[:, None]).sum(dim=-1)` — для каждого из 20 кандидатов считаем, сколько кодов из 4 совпало с таргетом. Результат `(B, 20)`.

> **Что с чем сравнивается:** покодово собранный кандидат (4 числа в диапазоне 0..255 каждый) vs. семантический ID ground-truth айтема (тоже 4 числа). Совпадение **всех 4 кодов** = «модель угадала айтем». Сам item_id при этом нигде явно не сравнивается — сравнение идёт через семантические ID. Поскольку `RQKmeansPipeline` с collision_solver выдаёт **уникальные** 4-кортежи кодов на айтем, эта эквивалентность 1-в-1.

---

## 5. CorrectItemsLogitsProcessor — почему предсказания «существуют» и не повторяют историю

[models.py:7-59](../../MastersDiploma/scripts/tiger/models.py#L7-L59). Конструктор:

```python
def __init__(self, num_codebooks, codebook_size, mapping, num_beams, visited_items):
    ...
    semantic_ids = []
    for i in range(len(mapping)):
        assert len(mapping[str(i)]) == num_codebooks, 'All semantic ids must have the same length'
        semantic_ids.append(mapping[str(i)])

    self.index_semantic_ids = torch.tensor(semantic_ids, dtype=torch.long, device=visited_items.device)
    # (num_items, semantic_ids)

    batch_size, _ = visited_items.shape
    self.index_semantic_ids = torch.tile(self.index_semantic_ids[None], dims=[batch_size, 1, 1])
    # (batch_size, num_items, semantic_ids)

    index = visited_items[..., None].tile(dims=[1, 1, num_codebooks])
    self.index_semantic_ids = torch.scatter(
        input=self.index_semantic_ids,
        dim=1,
        index=index,
        src=torch.zeros_like(index)
    )  # (batch_size, num_items, semantic_ids)
```

Главная идея: для каждого юзера в батче строится таблица «допустимых» семантических ID размером `(num_items, 4)`. Затем по `visited_items` (история этого юзера) посещённые позиции **обнуляются** — конкретно, их 4 кода переписываются нулями (в реальном вокабуляре код 0 принадлежит кодбуку 0, кстати — но для всех остальных кодбуков нули — это «вне диапазона», что дальше отсекается через `scores[:, :next_sid_codebook_num * codebook_size] = -inf`).

Затем в `__call__` ([models.py:32-59](../../MastersDiploma/scripts/tiger/models.py#L32-L59)):

```python
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    next_sid_codebook_num = (torch.minimum((input_ids[:, -1].max() // self.codebook_size), torch.as_tensor(self.num_codebooks - 1)).item() + 1) % self.num_codebooks
    a = torch.tile(self.index_semantic_ids[:, None, :, next_sid_codebook_num], dims=[1, self.num_beams, 1])
    a = a.reshape(a.shape[0] * a.shape[1], a.shape[2])

    if next_sid_codebook_num != 0:
        b = torch.tile(self.index_semantic_ids[:, None :, :next_sid_codebook_num], dims=[1, self.num_beams, 1, 1])
        b = b.reshape(b.shape[0] * b.shape[1], b.shape[2], b.shape[3])

        current_prefixes = input_ids[:, -next_sid_codebook_num:]
        possible_next_items_mask = (
            torch.eq(current_prefixes[:, None, :], b).long().sum(dim=-1) == next_sid_codebook_num
        )
        a[~possible_next_items_mask] = (next_sid_codebook_num + 1) * self.codebook_size

    scores_mask = torch.zeros_like(scores).bool()
    scores_mask = torch.scatter_add(
        input=scores_mask,
        dim=-1,
        index=a,
        src=torch.ones_like(a).bool()
    )

    scores[:, :next_sid_codebook_num * self.codebook_size] = -torch.inf
    scores[:, (next_sid_codebook_num + 1) * self.codebook_size:] = -torch.inf
    scores[~(scores_mask.bool())] = -torch.inf

    return scores
```

Что тут:
1. По последнему сгенерированному токену определяется **номер текущего кодбука** `next_sid_codebook_num` (на каком шаге мы находимся).
2. Сначала диапазон допустимых токенов сужается до сдвинутого кодбука: `[t * 256, (t+1) * 256)`.
3. Среди этих 256 кандидатов оставляются только те, чей **префикс из уже сгенерированных кодов** соответствует префиксу хотя бы одного существующего (и не посещённого) айтема. То есть на шаге t=2 из 256 возможных значений `c2` оставим только те, для которых существует айтем с уже сгенерированными `(c0, c1, c2)`.
4. Всё остальное → `-inf` → beam search никогда туда не пойдёт.

Это и есть «семантический идентификатор не сразу собирается». Beam search идёт пошагово: на t=0 выбирается код первого кодбука, на t=1 — второго (с учётом, что (c0, c1) — реальный префикс), и т.д. После 4 шагов имеем валидный полный ID существующего и непосещённого айтема.

> Из-за этого Recall@K по факту измеряется **по непосещённым айтемам** — модель физически не может предложить что-то, с чем юзер уже взаимодействовал. Сравните с классическим SASRec, где это нужно делать вручную через маскирование скоринга.

---

## 6. Сами числа: как из (B, 20, 4) получить Recall@K и NDCG@K

[models.py:174-193](../../MastersDiploma/scripts/tiger/models.py#L174-L193):

```python
all_hits = (torch.eq(predictions, labels[:, None]).sum(dim=-1))  # (batch_size, top_k=20)

target_item_ids = inputs['labels.ids'] if self._bucket_names else None
in_bucket = {
    name: getattr(self, f'_bucket_mask_{name}')[target_item_ids]
    for name in self._bucket_names
}

for k in [5, 10, 20]:
    hits = (all_hits[:, :k] == self._sem_id_len).float()  # (batch_size, k), 0/1 — полное совпадение
    recall = hits.sum(dim=-1)  # (batch_size,)
    discount_factor = 1 / torch.log2(torch.arange(1, k + 1, 1).float() + 1.).to(hits.device)  # (k,)
    ndcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size,)

    metrics[f'recall@{k}'] = recall.cpu().float()
    metrics[f'ndcg@{k}'] = ndcg.cpu().float()

    for name, mask in in_bucket.items():
        metrics[f'recall@{k}_{name}'] = recall[mask].cpu().float()
        metrics[f'ndcg@{k}_{name}'] = ndcg[mask].cpu().float()
```

Разбор по шагам:
- `all_hits[:, :k]` — берём топ-K из 20 кандидатов (они уже отсортированы beam search'ем по убыванию score).
- `hits = (all_hits[:, :k] == sem_id_len)` — `True` только если **все 4 кода** совпали → айтем угадан. Из-за уникальности семантических ID после collision_solver сумма по строке `hits` равна 0 или 1 (целевой айтем встречается в топ-K максимум один раз).
- `recall = hits.sum(dim=-1)` — это **HitRate@K** (он же Recall@K при единственном релевантном айтеме на юзера). Значение 0 или 1.
- `ndcg = sum_k hits[:, k] / log2(k + 2)` — стандартная формула DCG. Идеальный DCG для одного релевантного = `1 / log2(2) = 1`, поэтому **деление на ideal-DCG не нужно**, и DCG = NDCG численно. Это нормальная упрощённая форма для leave-one-out.

Финальный Mean-NDCG/Mean-Recall по всему датасету собирается через `cb.MeanAccumulator` в [train.py:177-205](../../MastersDiploma/scripts/tiger/train.py#L177-L205):

```python
# train.py:176-206
cb.Validation(
    dataset=valid_dataloder,
    callbacks=[
        cb.BatchMetrics(metrics=lambda model_outputs, _: {
            'loss': model_outputs['loss'].item(),
            **{name: model_outputs[name].tolist() for name in metric_names},
        }, name='validation'),
        cb.MetricAccumulator(
            accumulators={
                'validation/loss': cb.MeanAccumulator(),
                **{f'validation/{name}': cb.MeanAccumulator() for name in metric_names},
            },
        ),
    ],
).every_num_steps(EPOCH_NUM_STEPS),
```

Каждый батч даёт списки `recall@5`, `ndcg@5`, ... длины `B`; `MeanAccumulator` сливает их в один большой список и берёт среднее ([metrics.py:80-101](../../MastersDiploma/src/irec/callbacks/metrics.py#L80-L101)) — то есть это **взвешенное по примерам** среднее (а не наивное «среднее средних батчей»).

`cb.Validation` ([metrics.py:148-175](../../MastersDiploma/src/irec/callbacks/metrics.py#L148-L175)) — обёртка, которая создаёт `InferenceRunner` и пробегает по всему `valid_dataloder` / `eval_dataloder`. Это значит что валидация считается по **всему** валидационному/тестовому датасету, не по одному батчу.

---

## 7. Cold/Warm/Hot бакеты: чем расширены метрики

[train.py:62-92](../../MastersDiploma/scripts/tiger/train.py#L62-L92):

```python
inter_json_path = os.path.join(IREC_PATH, data_prefix + 'inter.json')
with open(inter_json_path, 'r') as f:
    user_interactions = json.load(f)

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
hot_mask = item_freqs >= 5

item_buckets = {'cold': cold_mask, 'warm': warm_mask, 'hot': hot_mask}
```

Что важно:
- Частоты считаются **только по `items[:-2]`** — train history. Это намеренно: иначе таргеты val/test «утекли» бы в пороги бакетов.
- Бакеты **накладываются** (hot ⊃ warm ⊃ cold) — это перекрывающиеся, не разбиение. Так что `recall@5_cold + recall@5_warm + recall@5_hot ≠ recall@5_total`.
- Бакетирование идёт **по таргет-айтему** ([models.py:176-180](../../MastersDiploma/scripts/tiger/models.py#L176-L180)):
  ```python
  target_item_ids = inputs['labels.ids'] if self._bucket_names else None
  in_bucket = {
      name: getattr(self, f'_bucket_mask_{name}')[target_item_ids]
      for name in self._bucket_names
  }
  ```
  То есть `recall@5_cold` — это средний recall **только по тем семплам, где ground-truth айтем редкий (5..10 встреч в train history)**.
- `labels.ids` — это **исходный item_id** таргета (не семантический), который проносится через всю цепочку трансформов как есть. Используется и для бакетной маски, и больше нигде.

---

## 8. Train vs validation vs eval — чем отличаются

| Этап | Что в `item.ids` | Что в `labels.ids` | Считаются метрики? |
|------|------------------|---------------------|--------------------|
| Train | `[i1..i_{N-2}]` (точнее, `[i1..i_{N-3}]` после `:-1` в `_last_item_transform`) | `[i_{N-2}]` | Только loss. Beam search не запускается (`self.training == True`). |
| Validation | `[i1..i_{N-1}][:-1]` = `[i1..i_{N-2}]` | `[i_{N-1}]` | Loss + полный набор recall@K, ndcg@K, бакетные. |
| Eval (test) | `[i1..i_N][:-1]` = `[i1..i_{N-1}]` | `[i_N]` | То же самое. |

> **Тонкая деталь:** на train можно включить `is_extended=True` (по умолчанию в `varka.py` оно `True`, см. [varka.py:210](../../MastersDiploma/scripts/tiger/varka.py#L210)), и тогда из истории генерируется не один, а много семплов — все префиксы `[i1..i_k]` для `k ∈ [2, N-2]`. Это data augmentation для train; на val/test всегда одна точка leave-one-out.

С точки зрения схемы предсказания train и eval **симметричны** (next-item), отличаются только:
1. На train `forward` идёт через teacher forcing (видим истинные семантические коды декодера),
2. На eval вместо teacher forcing — beam search с логитс-процессором,
3. На eval дополнительно фильтруются `visited_items` (на train этого не нужно — там и не дотягиваемся до beam search).

---

## 9. Что записывается в TensorBoard

[train.py:159-209](../../MastersDiploma/scripts/tiger/train.py#L159-L209) — `metric_names` строится так:

```python
metric_names = []
for k in (5, 10, 20):
    metric_names += [f'recall@{k}', f'ndcg@{k}']
    for bucket in ('cold', 'warm', 'hot'):
        metric_names += [f'recall@{k}_{bucket}', f'ndcg@{k}_{bucket}']
```

В тенсорборд уходят:
- `train/loss`,
- `validation/loss`, `validation/recall@K`, `validation/ndcg@K`, `validation/recall@K_{cold,warm,hot}`, `validation/ndcg@K_{cold,warm,hot}` для K∈{5,10,20},
- то же самое для `eval/...`.

EarlyStopping мониторит `eval/ndcg@20` ([train.py:212](../../MastersDiploma/scripts/tiger/train.py#L212)) — то есть «лучшая модель» = максимум NDCG@20 на тестовом сете. Сабкомментарий: формально использовать тест для early stopping — методологически нечисто (нужно бы valid/ndcg@K), но тут сделано так и менять, согласно ТЗ, нельзя.

---

## 10. Кто кого сравнивает: визуально

```
inter.json (user → [item_ids])
            │
            ▼
    Dataset.create()  ← leave-one-out на позициях
            │
            ├── train_dataset:    item.ids = history[:-2], labels.ids = history[-3]   (или, при extended, по всем префиксам)
            ├── valid_dataset:    item.ids = history[:-2], labels.ids = history[-2]
            └── eval_dataset:     item.ids = history[:-1], labels.ids = history[-1]
                                                                         ▲
                                                                         │   ground truth — именно сюда
                                                                         │
            ▼
    SemanticIdsMapper(index_rqkmeans.json)  ← item_id -> [c0,c1,c2,c3]
            │
            ▼
    TigerProcessing  ← prepend DECODER_START к labels, склейка user_hash к input
            │
            ▼  (это происходит в varka.py, оффлайн → arrow)
    батч в модель (.arrow)
            │
            ▼
    model.forward()
       ├── train: cross-entropy(decoder predicts [c0,c1,c2,c3]) → backprop
       └── eval:  beam search 30 → 20 кандидатов × 4 кода
                       │
                       │ в каждом шаге CorrectItemsLogitsProcessor:
                       │   - режет диапазон до текущего кодбука
                       │   - оставляет только префиксы существующих не-visited айтемов
                       ▼
                  predictions: (B, 20, 4)   labels: (B, 4)
                       │            │
                       └─── eq, sum по 4 ── all_hits: (B, 20) ∈ {0,1,2,3,4}
                                                │
                       ┌─────────────── (== 4) ┘
                       ▼
                hits[:, :K]: (B, K) ∈ {0,1}  ← полное совпадение all-or-nothing
                       │
                       ├─── sum    → recall@K = HR@K
                       └─── × 1/log2(rank+1), sum → ndcg@K (= dcg@K, ideal=1)
                       │
                       ▼
              MeanAccumulator → tensorboard
```

---

## 11. Резюме кратко

- **Ground truth** — единственный leave-one-out item_id на юзера. Через `index_rqkmeans.json` он превращается в кортеж из 4 кодов, и именно эти 4 кода служат «истинной меткой» для сравнения.
- **Predictions** — 20 лучших семантических ID, собранных beam search'ем покодово, при ограничениях логитс-процессора (только валидные коды текущего кодбука + префикс существующего не-visited айтема).
- **Recall@K** = доля примеров, в чьих топ-K кандидатов попал точный семантический ID таргета (т.е. все 4 кода совпали). Это то же самое, что **HR@K** — формулировки совпадают при единственном релевантном айтеме.
- **NDCG@K** = `Σ_{k=1..K} hit_k / log2(k + 1)`. Так как ideal-DCG = 1, это уже нормированная величина.
- **Бакеты cold/warm/hot** считаются по частоте таргет-айтема в train history (`items[:-2]`); пороги — `[5,10]`/`[5,20]`/`[5,∞]`. Бакеты вложены, не дизъюнктны.
- **EarlyStopping**: `eval/ndcg@20`, patience=40. Лучший чекпоинт = максимум этой метрики на test set'е.

---

## 12. Что это значит для VK-эксперимента

Главное для VK-плана — это то, что **оценочная схема не привязана к Amazon-специфике вообще нигде**:

- Никаких хардкодов на num_items, размер кодбуков и т.п. в логике метрик.
- `inter.json` обязан давать ≥5 айтемов на юзера ([data.py:45](../../MastersDiploma/scripts/tiger/data.py#L45) — assert), отсюда требование Core-5 в `vk_plan.md` (пункт 1 «подводных камней»).
- `index_rqkmeans.json` обязан содержать ключ для **каждого** item_id, встретившегося в `inter.json` ([varka.py:135](../../MastersDiploma/scripts/tiger/varka.py#L135) — assert) — отсюда требование согласованности маппинга.
- При том же `num_codebooks=4` и `codebook_size=256` все формулы метрик идентичны Amazon → VK-результаты будут напрямую сравнимы по recall@K/ndcg@K.
- Бакеты cold/warm/hot имеют фиксированные пороги частот (5/10/20). Для VK с другим распределением частот распределение по бакетам будет совершенно другим — это нормально, но пороги вшиты в `train.py` и тоже относятся к категории «менять можно (это не код тайгера, это ранер)».
