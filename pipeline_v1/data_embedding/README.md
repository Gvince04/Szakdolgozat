# MOMENT Embedding Generátor

Ez a script (`data_embedding_new.py`) arra szolgál, hogy a feldolgozott `headset.npy` idősoros adatfájlokból beágyazásokat (embeddingeket) generáljon a Meta/AutonLab által fejlesztett **MOMENT** alapmodell segítségével.

A program rekurzívan végigmegy a megadott bemeneti könyvtáron, minden `headset.npy` fájlt beolvas, és létrehozza a hozzá tartozó `headset_embedded.npy` fájlt a kimeneti könyvtárban, megőrizve az eredeti mappaszerkezetet.

## Követelmények

A futtatáshoz szükséges Python környezet reprodukálható a `MOMENT/environment_data/` mappában található `.yml` fájl használatával:
```bash
conda env create -f environment.yml
```
<small>(Megjegyzés: Ha a fájl neve más, vagy máshol állsz a terminálban, igazítsd hozzá az útvonalat.)</small>

Amennyiben ez nem sikerülne akkor a következő csomagokkal kéne manuálisan telepíteni egy új környezetet:
* Python 3.8+ (ajánlott: 3.11)
* `torch` (CUDA támogatással ajánlott)
* `momentfm`
* `numpy`
* `scikit-learn`
* `tqdm`

## Használat

A scriptet terminálból futtathatod paraméterek megadásával.

### Alapvető parancs
A legegyszerűbb futtatás, ha elfogadod az alapértelmezett útvonalakat (vagy ha a scriptben átírtad őket):
```bash
python data_embedding_new.py
```

### Egyedi könyvtárak megadása (Ajánlott)
Ha a mappáid máshol vannak, használd a flageket:
```bash
python data_embedding_new.py --input_dir "../zengo_preprocessed" --output_dir "../zengo_embedded"
```

## Argumentumok 
Az elérhető kapcsolók (flag-ek) részletes leírása:
| Argumentum | Leírás | Alapértelmezett Érték |
| :--- | :--- | :--- |
| `--input_dir` | A **bemeneti** főkönyvtár elérési útja. Ebben keresi a script a `headset.npy` fájlokat (alkönyvtárakban is). | `../zengo_preprocessed` |
| `--output_dir` | A **kimeneti** főkönyvtár. Ide menti az elkészült `headset_embedded.npy` fájlokat, tükrözve a bemeneti mappaszerkezetet. | `../zengo_embedded` |
| `--model_name` | A használni kívánt MOMENT modell HuggingFace azonosítója. | `AutonLab/MOMENT-1-large` |
| `--chunk_size` | A modell kontextus ablakának mérete (tokenek/időpontok száma). | `2048` |
| `--overwrite` | Ha ezt a kapcsolót használod, a script **felülírja** a már létező kimeneti fájlokat. Enélkül átugorja őket. | *Kikapcsolva* |
| `--help`, `-h` | Megjeleníti a súgót és kilép. | - |

További használati információ eléréséhez használd:
```bash
python data_embedding_new.py --help
```