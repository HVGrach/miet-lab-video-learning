# Geometry EDA Acceptance

Дата фиксации: 2026-06-02.

## Что закрывает пункт 4

- Построен frame-level manifest по всему cleaned-датасету: 24 170 RGB-изображений.
- Построен sample-level manifest для явных 8-кадровых примеров: 364 sample, 2 912 кадров.
- Сохранён главный plot-in-plot: `outputs/eda/aspect_ratio_with_height_inset.png`.
- Сохранены дополнительные графики: `height_distribution_by_class.png`, `width_height_scatter.png`.
- Сохранён contact sheet для ручной проверки geometry outliers: `geometry_outliers_contact_sheet.png`.

## Контракты данных

- `frame_manifest.csv` не содержит абсолютный `path`; используется только `rel_path`.
- `sample_manifest.csv` хранит `frame_paths` как JSON-массив из 8 относительных путей.
- `folder_8` означает только папку без prefixed-групп и ровно с 8 кадрами.
- `remaining_8` означает ровно 8 оставшихся кадров после выделения prefixed-групп.

## Актуальные числа

| Класс | Кадров | `folder_8` | `prefixed_8` | `remaining_8` |
|---|---:|---:|---:|---:|
| `inaction` | 4 605 | 4 | 64 | 0 |
| `move` | 3 901 | 18 | 31 | 0 |
| `work` | 15 664 | 23 | 222 | 2 |
| Итого | 24 170 | 45 | 317 | 2 |

- Глобально уникальных пар `(width, height)`: 15 938.
- Диапазон ширины: 29..421 px.
- Диапазон высоты: 44..534 px.
- Диапазон aspect ratio: 0.198..3.674.
- Найдено 3 786 geometry outliers для ручного аудита; основная причина — `small_width`.

## Решение для preprocessing

- Использовать resize с сохранением aspect ratio и padding/letterbox.
- Не использовать размеры, aspect ratio, имена файлов или пути как признаки модели.
- Применять пространственные и цветовые аугментации одинаково ко всем 8 кадрам одного sample.
- Длинные треки пока не входят в sample manifest; для них нужен отдельный stride/dilation/window sampling.

## Проверка

- Notebook `notebooks/04_geometry_eda.ipynb` выполнен без ошибок.
- Contract-assertions внутри notebook прошли.
- Команда `PYTHONPATH=src python3 -m unittest discover -s tests -v` проходит 13 тестов.
