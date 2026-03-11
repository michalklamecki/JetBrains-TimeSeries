# Project Context for an Agent

## Project goal
Celem projektu jest zbudowanie systemu, który przewiduje, czy incydent (anomaly) wystąpi w ciągu następnych H kroków czasowych na podstawie poprzednich W kroków jednej lub więcej metryk czasowych.

## Struktura repozytorium (ważne ścieżki)
- Notebook główny: `notebooks/solution.ipynb` — zawiera importy, wczytywanie danych, wizualizacje, funkcje do generowania okien i ekstrakcji cech.
- Dane: `dataset/data/data/train` i `dataset/data/data/test` — pliki `.npy` z telemetryką (np. jeden plik ma kształt `(2880, 25)`).
- Etykiety anomalii: `dataset/labeled_anomalies.csv` — zawiera `chan_id` i `anomaly_sequences` (listy par `[start, end]`).
- Agent config: `.agents/TODO.MD`, `.agents/Context.MD` (ten plik).

## Co już zaimplementowano w notebooku
- Wczytywanie `anomaly_df` i list plików `train_files`/`test_files`.
- Proste wykresy (pierwsza kolumna z każdego pliku) i wizualizacja anomalii dla wybranego kanału.
- Funkcje pomocnicze:
  - `ranges_overlap(a_s,a_e,b_s,b_e)` — sprawdza nakładanie zakresów.
  - `label_for_end(t_end, H, anomaly_ranges)` — zwraca 1 jeśli w przedziale przyszłym `[t_end+1, t_end+H]` jest anomalia.
  - `windows_labels_from_array(data, W, H, step=1, anomaly_ranges=None)` — generuje okna z przeszłości kończące się w `t_end` i odpowiadające etykiety (zwraca `X (N,W,C)`, `y`, `starts`).
  - `windows_to_feature_df(X_windows)` — ekstrakcja cech statystycznych (mean, std, min, max, range, slope) per-channel.

## Zalecane konwencje i ograniczenia
- Okno szeregu: definiowane jako `W` próbek z przeszłości kończących się na indeksie `t_end` (zawiera `[t_end-W+1 .. t_end]`).
- Etykieta: 1 jeśli w horyzoncie przyszłości `[t_end+1 .. t_end+H]` wystąpi choć jeden punkt anomalii.
- Unikaj przecieku danych: normalizacja (`StandardScaler`) powinna być dopasowana tylko na zbiorze treningowym.
- Pamięć: surowe pliki `.npy` są relatywnie małe; jednak stackowanie wszystkich okien może być pamięciożerne. Preferowane opcje:
  - Ekstrakcja cech per-file i zapis na dysk (Parquet/npz), potem trening klasycznych modeli na scalonych cechach.
  - Dla deep learning: użycie generatorów (`tf.data.Dataset` lub PyTorch `DataLoader`) z `mmap` lub lazy loading.
  - Konwersja do `float32` pozwala zaoszczędzić pamięć.

## Proponowane następne kroki (priorytety)
1. Agregacja okien z wszystkich plików treningowych i zapis cech per-file.
2. Skalowanie cech (fit `StandardScaler` na train), podział na train/val (time-based) i trening baseline RandomForest/LightGBM.
3. Przygotowanie generatora dla modeli sekwencyjnych i trening 1D-CNN/LSTM na surowych oknach.
4. Ewaluacja z metrykami alertowymi (precision/recall/F1, PR-AUC, detection latency).

## Jak uruchomić notebook (szybkie wskazówki dla agenta)
1. Upewnij się, że `base_path = '../dataset'` jest poprawna względem lokalizacji notebooka.
2. Uruchom komórkę z importami (Cell 1) i komórki wczytujące `anomaly_df` oraz `train_files` (Cell 2–5).
3. Uruchom komórkę z funkcjami `windows_labels_from_array` i `windows_to_feature_df` (już dodane w notebooku, około Cell 9–12).
4. Do generacji danych: dla każdego pliku załaduj `.npy`, sparsuj `anomaly_sequences` z `anomaly_df` (użyj `ast.literal_eval`), wywołaj `windows_labels_from_array`, a następnie `windows_to_feature_df` jeśli chcesz cechy.

## Zasoby i zależności
- Wymagane pakiety: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow` (opcjonalnie) lub `torch` (opcjonalnie).
- Notebook ma przykładowe funkcje i wizualizacje gotowe do użycia.

## Kontakt i oczekiwania
- Agent powinien generować cechy per-file i tworzyć scalony dataset cech oraz zapisać go w `artifacts/` lub `data/processed/` przed treningiem.
- Agent może zaproponować wartości domyślne `W` i `H` (np. W=50, H=10) i wykonać szybki baseline, a następnie rozszerzyć eksperymenty.

---
Plik wygenerowany na podstawie historii rozmowy i aktualnego stanu `notebooks/solution.ipynb` (funkcje do okien i ekstrakcji cech są już w notebooku).