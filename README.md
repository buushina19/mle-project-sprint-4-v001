# Подготовка виртуальной машины

## Склонируйте репозиторий

```
git clone <ссылка-на-ваш-репозиторий>
cd mle-project-sprint-4-v001
```

## Активируйте виртуальное окружение

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Скачайте файлы с данными

```
mkdir -p data
wget -P data/ https://storage.yandexcloud.net/mle-data/ym/tracks.parquet
wget -P data/ https://storage.yandexcloud.net/mle-data/ym/catalog_names.parquet
wget -P data/ https://storage.yandexcloud.net/mle-data/ym/interactions.parquet
```

Создайте файл `.env`:

```
S3_BUCKET_NAME=...
S3_ENDPOINT_URL=https://storage.yandexcloud.net
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

## Запустите Jupyter Lab

```
jupyter lab --ip=0.0.0.0 --no-browser
```

# Расчёт рекомендаций

Основной код — в `recommendations.ipynb`.

Полный пайплайн (этапы 1–3) можно запустить одной командой:

```
python scripts/build_recommendations.py
```

Скрипт создаёт `items.parquet`, `events.parquet`, `top_popular.parquet`, `personal_als.parquet`, `similar.parquet`, `recommendations.parquet` и загружает их в S3 (`recsys/data/` и `recsys/recommendations/`).

# Сервис рекомендаций (этап 4)

Код сервиса — в `recommendations_service.py`. Перед запуском должны лежать в корне проекта:

- `recommendations.parquet`, `top_popular.parquet`, `similar.parquet`, `events.parquet`

## Запуск сервиса

```bash
source venv/bin/activate
uvicorn recommendations_service:app --host 127.0.0.1 --port 8000
```

Проверка: откройте в браузере `http://127.0.0.1:8000/` — должно вернуться `{"message": "Recommendations service is working"}`.

### Эндпоинты

| Метод | URL | Назначение |
|-------|-----|------------|
| POST | `/events/put?user_id=&item_id=` | добавить трек в онлайн-историю пользователя |
| POST | `/recommendations_offline?user_id=&k=` | офлайн-рекомендации с учётом истории прослушиваний |
| POST | `/recommendations_online?user_id=&k=` | онлайн-рекомендации по последним 3 онлайн-событиям |
| POST | `/recommendations?user_id=&k=` | смешанные рекомендации |

## Стратегия смешивания онлайн- и офлайн-рекомендаций

1. **Офлайн-часть** — персональные рекомендации из `recommendations.parquet` (CatBoost). Если пользователя нет в файле, отдаём `top_popular.parquet`.
2. **Онлайн-часть** — по 3 последним онлайн-событиям пользователя (in-memory store) ищем похожие треки в `similar.parquet`, сортируем по score, убираем дубликаты.
3. **История** — из `events.parquet` (офлайн) и онлайн-событий исключаем уже прослушанные треки.
4. **Смешивание** — в итоговом списке на **чётных** позициях (0, 2, 4, …) стоят офлайн-рекомендации, на **нечётных** (1, 3, 5, …) — онлайн. Оставшиеся элементы дописываются в конец, затем список дедуплицируется и обрезается до `k`.

# Тестирование сервиса

Код тестов — в `test_service.py`. Вывод сохраняется в `test_service.log`.

В **первом** терминале запустите сервис (см. выше). Во **втором**:

```bash
source venv/bin/activate
python test_service.py
```

Скрипт проверяет три сценария:

1. `user_id=0` — нет персональных рекомендаций, fallback на топ популярных;
2. `user_id=4` — есть персональные офлайн-рекомендации, онлайн-история пуста;
3. `user_id=4` — добавляются онлайн-события, смешанные рекомендации отличаются от чисто офлайн.
