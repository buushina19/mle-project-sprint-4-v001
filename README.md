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

# Сервис рекомендаций

Код сервиса рекомендаций находится в файле `recommendations_service.py`.

<*укажите здесь необходимые шаги для запуска сервиса рекомендаций*>

# Инструкции для тестирования сервиса

Код для тестирования сервиса находится в файле `test_service.py`.

<*укажите здесь необходимые шаги для тестирования сервиса рекомендаций*>
