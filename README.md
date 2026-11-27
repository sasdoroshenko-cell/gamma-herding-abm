# Хеджирование опционов как скрытая связь

Финальный проект по финансовой математике (ВШЭ, ФКН): как гамма дилеров опционов меняет фазовый переход стадности в агентной модели рынка с книгой заявок.

База — агентная модель из курсовой (гипотеза 1, ветка `hypothesis-1-phase-transition`).

## Окружение

Нужен Python 3.11 или новее. На macOS 27 scipy для Python 3.10 не загружается (ошибка `_spropack ... zero-fill section type`), поэтому старое окружение курсовой не подходит.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

На Windows: `.venv\Scripts\pip install -r requirements.txt`. Версии в `requirements.txt` закреплены и ставятся готовыми колёсами на Python 3.11–3.14 (macOS arm64 и x86_64, Windows, Linux), ничего не компилируется. Если подходящего Python нет, его можно поставить через `uv` (`pip install uv`, затем `uv python install 3.12` и `uv venv --python 3.12 .venv`) или с python.org.

## VS Code

1. File → Open Folder → эта папка (`1d-abm-phase-transition`).
2. Cmd+Shift+P → Python: Select Interpreter → `./.venv/bin/python`.
3. Тесты: вкладка Testing (колба) или `pytest` в терминале.

## Запуск

```bash
python scripts/smoke_run.py --J 0          # один прогон, сводка и график цены и |m|
python scripts/smoke_run.py --J 0.8 --a1 0.6
python scripts/smoke_run.py --J 0.5 --a1 0.6 --dealer short --book 288   # с дилерами
python scripts/debug_sweep_J.py --quick    # маленькая развёртка по J, проверка что всё работает
python scripts/debug_sweep_J.py            # отладочная развёртка: docs/debug_sweep_J.png
python scripts/mechanism_check.py          # проверка механизма хеджа: docs/mechanism_check.png
pytest                                     # проверки модели
```

## Дилеры

12 дилеров опционов (`AgentBasedModel/agents/dealer.py`) есть в каждом рынке. Режим задаётся в `MarketConfig`:

```python
MarketConfig(dealer_side='short', dealer_book=288)   # продали 288 стрэддлов на всех (короткая гамма)
MarketConfig(dealer_side='long', dealer_book=288)    # купили (длинная гамма)
MarketConfig(dealer_side='none')                     # книга пуста, дилеры не торгуют
```

Каждый дилер держит лестницу стрэддлов около денег (новый каждые 15 шагов, срок 60, закрытие за 5 шагов до экспирации), каждый шаг считает дельту и гамму книги по Блэку–Шоулзу и рыночной заявкой держит позицию дельта-нейтральной. Журнал дилера — `dealer.history`; гамма всей книги по шагам — `dealer_gamma(market)`, поток дилеров — `dealer_flow(market)`. Размер книги 288 — временный, для отладки; окончательный n₀ считается по правилу из prereg после калибровки.

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

## Эксперименты и анализ

Все параметры прогонов — в `experiments/config.py`: сетка J, длина прогона и burn-in, число репликаций, размеры книги, параметры гистерезиса и анализа. Сид каждого прогона — функция от (конфигурация, размер книги, точка сетки, репликация) по формуле из prereg, поэтому любой прогон повторяется, а счёт можно делить между ноутбуками и потом сливать CSV.

n₀, J_fix и J_ref считаются по правилам prereg после калибровки и пока не заданы: полномасштабные прогоны без них не запускаются. Отладочные прогоны идут с предварительными значениями и отдельным потоком сидов.

```bash
python experiments/run_sweep.py --scale debug --reps 2 --grid-step 0.5 --steps 600 --burn-in 200   # проверка
python experiments/run_sweep.py --scale main            # основной эксперимент: 3 × 21 × 60 прогонов
python experiments/run_sweep.py --scale sensitivity     # + ещё 4 размера книги по 30 репликаций
python experiments/run_sweep.py --scale main --configs short   # только часть, например на другом ноутбуке
python experiments/run_hysteresis.py --scale main       # петли гистерезиса для В2
python experiments/pilot.py                             # пилот всего конвейера и замер времени: docs/pilot.md
python experiments/power_check.py                       # мощность главной гипотезы без дилеров: docs/power.md
python experiments/kappa_check.py                       # механизм и устойчивость с дилерами: docs/kappa_check.md
python analysis/make_figures.py                         # рисунки F4–F8 из results/summary/
```

Каждый прогон записывается в CSV сразу после окончания (`results/summary/sweep.csv`, `results/summary/hysteresis.csv`), повторный запуск пропускает готовое. Прерванные прогоны записываются с шагом остановки, а недостающие репликации добавляются по правилам замены из prereg (раздел 7).

Анализ:

```python
from analysis import susceptibility as sus, stats_tests as st

df = sus.load_sweep('results/summary/sweep.csv')
jc = sus.jc_table(df)                       # J_c каждой репликации: argmax χ(J) + парабола, флаг края
report = st.main_hypothesis_test({c: jc[jc.config == c].jc for c in ('short', 'none', 'long')})
print(report.to_markdown())                 # H0/H1, статистика, p, размеры эффекта с ДИ, вывод
```

## Структура

| Путь | Что | Кто |
|------|-----|-----|
| `AgentBasedModel/` | ядро модели: стакан, агенты, симулятор | A |
| `AgentBasedModel/agents/dealer.py` | дилер опционов: лестница стрэддлов и дельта-хедж | A |
| `AgentBasedModel/builder.py` | сборка рынка проекта: 112 агентов (из них 12 дилеров), кольцо чартистов, параметры | A |
| `options/` | Блэк–Шоулз (цена, дельта, гамма колла, пута, стрэддла) и историческая волатильность для дилера | A |
| `scripts/` | пробный запуск, отладочная развёртка, проверка механизма | A |
| `experiments/` | параметры и сиды, развёртки по J, гистерезис, пилот конвейера | C |
| `analysis/` | χ(J) и J_c, тесты (Джонкхир–Терпстра, Манн–Уитни, Холм, бутстрэп), рисунки | C |
| `results/summary/` | сводные CSV полных прогонов (в git) | C |
| `tests/` | тесты | все, каждый к своим модулям |
| `docs/` | prereg, отладочные рисунки, пилот | все |

Дальше по плану появятся `calibration/` и `frey/` (B).

