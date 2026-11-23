# Хеджирование опционов как скрытая связь

Финальный проект по финансовой математике (ВШЭ, ФКН): как гамма дилеров опционов меняет фазовый переход стадности в агентной модели рынка с книгой заявок.

База — агентная модель из курсовой (гипотеза 1, ветка `hypothesis-1-phase-transition`).

## Окружение

Нужен Python 3.11 или новее. На macOS 27 scipy для Python 3.10 не загружается (ошибка `_spropack ... zero-fill section type`), поэтому старое окружение курсовой не подходит.

```bash
python3.12 -m venv .venv
.venv/bin/pip install numpy scipy pandas matplotlib statsmodels tqdm seaborn pytest
```

Если Python 3.12 нет, его можно поставить через `uv` (`pip install uv`, затем `uv python install 3.12` и `uv venv --python 3.12 .venv`) или с python.org. Точные версии пакетов зафиксирует `requirements.txt` (задача C-1).

## VS Code

1. File → Open Folder → эта папка (`1d-abm-phase-transition`).
2. Cmd+Shift+P → Python: Select Interpreter → `./.venv/bin/python`.
3. Тесты: вкладка Testing (колба) или `pytest` в терминале.

## Запуск

```bash
python scripts/smoke_run.py --J 0          # один прогон, сводка и график цены и |m|
python scripts/smoke_run.py --J 0.8 --a1 0.6
python scripts/debug_sweep_J.py --quick    # маленькая развёртка по J, проверка что всё работает
python scripts/debug_sweep_J.py            # отладочная развёртка: docs/debug_sweep_J.png
pytest                                     # проверки изменений модели
```

## Журнал потока заявок

На каждом шаге биржа записывает «агрессивный» поток: сколько каждый тип агентов купил (+) или продал (−) заявками, которые сразу исполнились против стакана (рыночные заявки и лимитные, пересёкшие спред). Заявки, которые просто встали в стакан, в поток не входят.

```python
from AgentBasedModel.builder import MarketConfig, build_market, run

market = build_market(MarketConfig(a1=0.6), seed=1)
run(market, 1500)
info = market.simulator.info
info.flows[t]                 # {'Random': 3, 'Chartist': -5, ...} за шаг t
info.prices[t + 1] - info.prices[t]   # изменение цены, которое этот поток вызвал
```

## Структура

| Путь | Что | Кто |
|------|-----|-----|
| `AgentBasedModel/` | ядро модели: стакан, агенты, симулятор | A |
| `AgentBasedModel/builder.py` | сборка рынка проекта: 100 агентов, кольцо чартистов, параметры | A |
| `scripts/` | пробный запуск и отладочная развёртка | A |
| `tests/` | тесты | все, каждый к своим модулям |
| `docs/` | prereg и отладочные рисунки | все |

Дальше по плану появятся `options/` и дилер (A), `calibration/` и `frey/` (B), `experiments/` и `analysis/` (C).

