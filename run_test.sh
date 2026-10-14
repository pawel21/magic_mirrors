#!/usr/bin/env bash
# Test "kwiecień jako baseline, maj jako test". Uruchamiaj z katalogu magic_mirrors/:
#   bash run_test.sh
# Po zmianie samego configu (bez liczenia cech od nowa):
#   SKIP_EXTRACT=1 bash run_test.sh
set -euo pipefail

FRAMES="/media/pgliwny/ADATA HD3304/Computer_Vision_system/data/MAGIC/webcam/test_img"                 # katalog z kadrami (kwiecień + maj, mogą być w podkatalogach)
OUT=wyniki/kwiecien_maj            # katalog wyników tego przebiegu
LABELS=input/labels/labels.csv           # z aplikacji do etykietowania
REVIEWED=input/labels/reviewed.csv
CFG=config.yaml
SKIP_EXTRACT=${SKIP_EXTRACT:-0}

#echo "== 0. testy jednostkowe"
# python -m pytest -q tests

if [ "$SKIP_EXTRACT" != "1" ]; then
  echo "== 1. selekcja kadrów"
  python scripts/01_qc.py "$FRAMES" "$OUT" "$CFG"
  echo "== 2. cechy luster (najdłuższy krok)"
  python scripts/02_extract.py "$OUT" "$CFG"
fi

echo "== 3. wynik anomalii"
python scripts/03_score.py "$OUT" "$CFG"

echo "== 4. referencje w dniach testowych"
python scripts/check_refs.py "$OUT"

echo "== 5. mapa luster"
python scripts/05_map.py "$OUT" max 15

if [ -f "$LABELS" ] && [ -f "$REVIEWED" ]; then
  echo "== 6. ocena na etykietach"
  python scripts/06_evaluate.py "$OUT" "$LABELS" "$REVIEWED" test
else
  echo "== 6. pominięta: brak $LABELS lub $REVIEWED"
fi

echo "Gotowe. Wyniki w $OUT"gru