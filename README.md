# magic_mirrors

Monitoring 249 luster MAGIC-II na kadrach z webcamu (830x480, pozycja parkingowa).

## Instalacja

    python -m pip install -e .

## Użycie

    python scripts/00_segment.py kadr.jpg wyniki/seg      # poligony, maska etykiet, nakładka
    python scripts/01_qc.py  /sciezka/do/kadrow wyniki    # selekcja kadrów -> qc.sqlite
    python scripts/02_extract.py wyniki                   # cechy -> X.npz (N, 249, F)
    python scripts/03_score.py   wyniki                   # wynik anomalii -> scores.parquet
    python scripts/04_benchmark.py wyniki                 # anomalie syntetyczne -> benchmark.csv
    python scripts/05_map.py wyniki [wariant]             # mapa luster wg odsetka alarmów
    python scripts/06_evaluate.py wyniki labels.csv reviewed.csv [test]   # ocena na etykietach
    python -m pytest tests

Kadry mogą leżeć w dowolnej strukturze katalogów; czas UTC jest czytany z nazwy pliku
(`YYYYMMDD_HHMM`, wzorzec w `config.yaml`). Skrypty uruchamiaj z katalogu głównego pakietu.

## Moduły

| moduł | rola |
|---|---|
| `geometry.py` | model 249 luster: poligony, maski, rejestr idx/row/col |
| `frames.py` | wczytanie kadru, czas z nazwy, położenie Słońca |
| `qc.py` | selekcja kadrów, miary + powód odrzucenia |
| `features.py` | cechy per lustro (słownik `FEATURES`) |
| `baseline.py` | referencje po kącie Słońca, normalizacja w strefie, baseline lustra |
| `scoring.py` | warianty wyniku (`VARIANTS`), próg z kalibracji, veto, trwałość |
| `inject.py` | anomalie syntetyczne: defocus, missing, swap |
| `store.py` | zapis/odczyt plików pośrednich, podział chronologiczny |

Nową cechę dodaje się jako funkcję w `FEATURES` i nazwę w `config.yaml`; nowy wariant
wyniku jako funkcję w `VARIANTS`.

## Co nie jest skalibrowane

- Progi QC i położenia ROI w `config.yaml` są wartościami startowymi z 6 kadrów.
- Parametry geometrii (`geometry.PARAMS`) dotyczą ustawienia kamery z kwietnia-maja 2024.
- Brak cechy `ncc_max` (wymaga pamięci podręcznej patchy referencyjnych).

## Test na zamrożonym baseline (np. kwiecień -> maj)

W `config.yaml` ustaw `baseline.ref_until` (ostatni dzień, z którego wolno brać referencje) oraz
zakresy dat w `split`. Po zmianie wystarczy powtórzyć krok 3 (i 6):

    python scripts/03_score.py wyniki config.yaml
